# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
from torch import nn
from torch.distributed._tensor import DTensor, Shard
from torch.distributed._tensor.placement_types import Partial
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)

from torchao.core.config import AOBaseConfig
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
    triton_fp8_gemm_1x128_128x1,
    triton_fp8_gemm_1x128_128x128,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import is_sm_at_least_90


class fp8_blockwise_mm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, block_size, out_dtype=torch.bfloat16, use_triton=False):
        assert block_size == 128, "Only support block_size=128"

        # Handle DTensor inputs - extract local tensors for triton kernels.
        # When weight is a DTensor (e.g. after parallelize_module for TP), we extract
        # the local tensor shard and record the DTensor metadata for backward.
        x_is_dtensor = isinstance(x, DTensor)
        weight_is_dtensor = isinstance(weight, DTensor)
        ctx.x_is_dtensor = x_is_dtensor
        ctx.weight_is_dtensor = weight_is_dtensor

        if x_is_dtensor:
            ctx.x_device_mesh = x.device_mesh
            ctx.x_placements = x.placements
            x = x.to_local()

        if weight_is_dtensor:
            ctx.weight_device_mesh = weight.device_mesh
            ctx.weight_placements = weight.placements
            weight = weight.to_local()

        # Temporarily reshape x to 2D tensor
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        # Cast inputs to fp8 blockwise using (1, block_size) scaling granularity in row major format.
        x_fp8, x_scale = triton_fp8_blockwise_act_quant_lhs(x, block_size)

        # Cast weight to fp8 blockwise using (block_size, block_size) scaling granularity, with transposed dims in column major format.
        weight_t_fp8, weight_t_scale = triton_fp8_blockwise_weight_quant_transposed_rhs(
            weight,
            block_size=block_size,
        )

        # out = input @ weight.T
        fp8_gemm = triton_fp8_gemm_1x128_128x128 if use_triton else torch._scaled_mm
        out = fp8_gemm(
            x_fp8,
            weight_t_fp8,
            x_scale,
            weight_t_scale,
            out_dtype=out_dtype,
        )
        out = out.reshape(*x_orig_shape[:-1], out.shape[-1])
        ctx.save_for_backward(x, weight)
        ctx.block_size = block_size
        ctx.out_dtype = out_dtype
        ctx.use_triton = use_triton

        # When weight is a DTensor, wrap the local output as a DTensor so that
        # _prepare_output_fn can redistribute it correctly.
        #   - ColwiseParallel (weight Shard(0)): each rank computed a column-slice of
        #     the output, so output placement is Shard(-1).
        #   - RowwiseParallel (weight Shard(1)): each rank computed a partial sum over
        #     the input features, so output placement is Partial() (needs all-reduce).
        if weight_is_dtensor:
            weight_placements = ctx.weight_placements
            if any(isinstance(p, Shard) and p.dim == 0 for p in weight_placements):
                out_placements = (Shard(-1),)
            else:
                out_placements = (Partial(),)
            out = DTensor.from_local(
                out, ctx.weight_device_mesh, out_placements, run_check=False
            )

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        block_size = ctx.block_size
        out_dtype = ctx.out_dtype
        use_triton = ctx.use_triton
        x_is_dtensor = ctx.x_is_dtensor
        weight_is_dtensor = ctx.weight_is_dtensor

        # Extract local tensor from DTensor grad_output.
        # For ColwiseParallel the autograd engine provides DTensor(Shard(-1)).
        # For RowwiseParallel it provides DTensor(Replicate) (gradient after allreduce).
        if isinstance(grad_output, DTensor):
            grad_output = grad_output.to_local()

        # Reshape input to 2D
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        # Reshape grad_output to 2D
        grad_output_orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, grad_output_orig_shape[-1]).contiguous()
        assert grad_output.shape[1] % 128 == 0, "unsupported"

        # Cast grad_output to fp8 blockwise 1x128 since it is the grad of the output activation.
        grad_output_fp8, grad_output_scale = triton_fp8_blockwise_act_quant_lhs(
            grad_output,
            block_size,
        )

        # Cast weight to fp8 blockwise to 128x128 in column major format.
        weight_fp8, weight_scale = triton_fp8_blockwise_weight_quant_rhs(
            weight,
            block_size=block_size,
        )

        # grad_x = grad_output @ weight
        fp8_gemm_1x128_128x128 = (
            triton_fp8_gemm_1x128_128x128 if use_triton else torch._scaled_mm
        )
        grad_x = fp8_gemm_1x128_128x128(
            grad_output_fp8,
            weight_fp8,
            grad_output_scale,
            weight_scale,
            out_dtype=out_dtype,
        )

        # Cast grad_output_t to fp8 blockwise with (1 x block_size) scaling groups, since it is
        # the grad of the output activation.
        # Write directly with transposed dims in row major format, as needed for dW calc.
        grad_output_t_fp8, grad_output_t_scale = (
            triton_fp8_blockwise_act_quant_transposed_lhs(
                grad_output,
                block_size,
            )
        )

        # Cast x to fp8 blockwise with (block_size x 1) scaling groups, in column major format.
        # RHS should have groupwise scales calculated colwise, so scaling groups do not cross the
        # contracting (K) dim.
        x_fp8, x_scale = triton_fp8_blockwise_act_quant_rhs(x, block_size)

        # grad_weight = grad_output.T @ x
        fp8_gemm_1x128_128x1 = (
            triton_fp8_gemm_1x128_128x1 if use_triton else torch._scaled_mm
        )
        grad_weight = fp8_gemm_1x128_128x1(
            grad_output_t_fp8,
            x_fp8,
            grad_output_t_scale,
            x_scale,
            out_dtype=out_dtype,
        )

        # Reshape grad_x to expected potentially 3D+ shape
        grad_x = grad_x.reshape(*grad_output_orig_shape[:-1], grad_x.shape[-1])

        # Reconstruct DTensor gradients to match the DTensor inputs from forward.
        if weight_is_dtensor:
            weight_placements = ctx.weight_placements
            weight_device_mesh = ctx.weight_device_mesh

            # ColwiseParallel (weight Shard(0)): each rank computed a partial grad_x
            # (over the column-slice of weight it owns), so we all-reduce to get the
            # full gradient before wrapping as DTensor(same placement as x).
            if any(isinstance(p, Shard) and p.dim == 0 for p in weight_placements):
                torch.distributed.all_reduce(
                    grad_x,
                    op=torch.distributed.ReduceOp.SUM,
                    group=weight_device_mesh.get_group(0),
                )

            # grad_weight is the gradient for the local weight shard, so it shares
            # the same placement as the weight DTensor.
            grad_weight = DTensor.from_local(
                grad_weight, weight_device_mesh, weight_placements, run_check=False
            )

        if x_is_dtensor:
            # Wrap grad_x as a DTensor with the same placement as the input x.
            # For ColwiseParallel: x was Replicate, grad_x is now full (after all-reduce).
            # For RowwiseParallel: x was Shard(-1), grad_x is the gradient for x's shard.
            grad_x = DTensor.from_local(
                grad_x, ctx.x_device_mesh, ctx.x_placements, run_check=False
            )

        return grad_x, grad_weight, None, None, None


class Float8BlockwiseLinear(nn.Linear):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
        dtype (torch.dtype): Data type for the weights. Defaults to torch.float8_e4m3fn.
    """

    supported_dtypes = [
        torch.bfloat16,
    ]

    def __init__(
        self,
        *args,
        block_size: int = 128,
        dtype=torch.bfloat16,
        use_triton=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert dtype in self.supported_dtypes, (
            f"Unsupported dtype: {dtype}. Supported dtypes: {self.supported_dtypes}"
        )
        assert is_sm_at_least_90(), "Only support SM90"
        self.block_size = block_size
        self.dtype = dtype
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return fp8_blockwise_mm.apply(
            x, self.weight, self.block_size, self.dtype, self.use_triton
        )

    @classmethod
    def from_float(
        cls,
        mod,
    ):
        assert mod.bias is None, "unsupported"
        assert mod.in_features % 128 == 0, "unsupported"
        assert mod.out_features % 128 == 0, "unsupported"
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8BlockwiseLinearConfig(AOBaseConfig):
    pass


@register_quantize_module_handler(Float8BlockwiseLinearConfig)
def _float8_blockwise_transform(module, config):
    return Float8BlockwiseLinear.from_float(module)


class Float8BlockwiseColwiseParallel(ColwiseParallel):
    """
    Tensor parallel ColwiseParallel for Float8BlockwiseLinear.

    Unlike Float8ColwiseParallel, this doesn't handle FP8 casting in the
    input/output preparation functions since Float8BlockwiseLinear performs
    FP8 casting internally within fp8_blockwise_mm.

    The weight is sharded on dim 0 (output features). Each rank receives the
    full input (Replicate) and produces a column-slice of the output (Shard(-1)).
    fp8_blockwise_mm detects the DTensor weight and wraps its output as
    DTensor(Shard(-1)) so _prepare_output_fn can redistribute it.

    Example usage::

        model = Float8BlockwiseLinear(in_features, out_features, bias=False)
        parallelize_module(model, device_mesh, Float8BlockwiseColwiseParallel())
    """

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a DTensor(Shard(-1)) set by fp8_blockwise_mm forward.
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, Float8BlockwiseLinear):
            raise ValueError(
                f"Expecting module to be Float8BlockwiseLinear but found {type(module)}"
            )
        return super()._apply(module, device_mesh)


class Float8BlockwiseRowwiseParallel(RowwiseParallel):
    """
    Tensor parallel RowwiseParallel for Float8BlockwiseLinear.

    Unlike Float8RowwiseParallel, this doesn't handle FP8 casting in the
    input/output preparation functions since Float8BlockwiseLinear performs
    FP8 casting internally within fp8_blockwise_mm.

    The weight is sharded on dim 1 (input features). Each rank receives a
    column-slice of the input (Shard(-1)) and produces a partial output sum.
    fp8_blockwise_mm detects the DTensor weight and wraps its output as
    DTensor(Partial()), which _prepare_output_fn reduces to Replicate via
    an all-reduce.

    Example usage (standalone, replicated input)::

        model = Float8BlockwiseLinear(in_features, out_features, bias=False)
        parallelize_module(
            model, device_mesh,
            Float8BlockwiseRowwiseParallel(input_layouts=(Replicate(),))
        )

    Example usage (after a ColwiseParallel layer, sharded input)::

        parallelize_module(
            model, device_mesh,
            {"fc1": Float8BlockwiseColwiseParallel(),
             "fc2": Float8BlockwiseRowwiseParallel()}
        )
    """

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a DTensor(Partial()) set by fp8_blockwise_mm forward.
        # Redistributing Partial() -> Replicate() triggers the all-reduce.
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, Float8BlockwiseLinear):
            raise ValueError(
                f"Expecting module to be Float8BlockwiseLinear but found {type(module)}"
            )
        return super()._apply(module, device_mesh)

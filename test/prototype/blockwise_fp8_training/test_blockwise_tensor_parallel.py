# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test tensor parallelism for Float8BlockwiseLinear.

To run this test:
    torchrun --nproc_per_node=2 test/prototype/blockwise_fp8_training/test_blockwise_tensor_parallel.py
"""

import copy
import os

import pytest
import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from tqdm import tqdm

from torchao.utils import is_sm_at_least_90

triton = pytest.importorskip("triton", reason="Triton required to run this test")
if not is_sm_at_least_90():
    pytest.skip("This test requires SM90 or higher", allow_module_level=True)


from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8_training.linear import (
    Float8BlockwiseColwiseParallel,
    Float8BlockwiseLinear,
    Float8BlockwiseRowwiseParallel,
)

torch.set_float32_matmul_precision("high")


def setup_distributed():
    """Setup distributed environment for testing"""
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    if world_size < 2:
        raise RuntimeError(
            "This test requires at least 2 GPUs. "
            "Run with: torchrun --nproc_per_node=2 test_blockwise_tensor_parallel.py"
        )
    device_mesh = init_device_mesh("cuda", (world_size,))
    # seed must be the same in all processes
    torch.manual_seed(42)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    return device_mesh


class SimpleModel(nn.Module):
    """Simple model with two linear layers for testing"""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


def _test_colwise_parallel_forward_backward(device_mesh, size=256):
    """Test ColwiseParallel forward and backward pass.

    Weight is sharded on dim 0 (output features). Input is replicated.
    Each rank computes a column-slice of the output (Shard(-1)).
    """
    in_features = size
    out_features = size
    batch_size = 4

    # Create reference model (non-parallel)
    ref_model = nn.Linear(in_features, out_features, bias=False).cuda()
    ref_blockwise = Float8BlockwiseLinear.from_float(copy.deepcopy(ref_model))

    # Create parallel model
    parallel_model = nn.Linear(in_features, out_features, bias=False).cuda()
    parallel_model.load_state_dict(ref_model.state_dict())
    parallel_blockwise = Float8BlockwiseLinear.from_float(parallel_model)

    # Apply column-wise parallelism at the module level.
    parallel_blockwise = parallelize_module(
        parallel_blockwise,
        device_mesh,
        Float8BlockwiseColwiseParallel(),
    )

    x = torch.randn(batch_size, in_features, device="cuda", requires_grad=True)
    x_parallel = x.clone().detach().requires_grad_(True)

    # Forward
    out_ref = ref_blockwise(x)
    out_parallel = parallel_blockwise(x_parallel)

    # Gather parallel output for comparison
    if isinstance(out_parallel, DTensor):
        out_parallel_gathered = out_parallel.redistribute(
            placements=[Replicate()]
        ).to_local()
    else:
        out_parallel_gathered = out_parallel

    sqnr = compute_error(out_ref, out_parallel_gathered)
    assert sqnr >= 25.0, f"Forward SQNR: {sqnr.item()} must be >= 25.0"

    # Backward pass
    out_ref.sum().backward()
    out_parallel.sum().backward()

    if isinstance(x_parallel.grad, DTensor):
        x_parallel_grad = x_parallel.grad.redistribute(
            placements=[Replicate()]
        ).to_local()
    else:
        x_parallel_grad = x_parallel.grad

    sqnr_bwd = compute_error(x.grad, x_parallel_grad)
    assert sqnr_bwd >= 25.0, f"Backward SQNR: {sqnr_bwd.item()} must be >= 25.0"

    print(
        f"✓ ColwiseParallel test passed (fwd SQNR: {sqnr:.2f}, bwd SQNR: {sqnr_bwd:.2f})"
    )


def _test_rowwise_parallel_forward_backward(device_mesh, size=256):
    """Test RowwiseParallel forward and backward pass.

    Uses input_layouts=(Replicate(),) since the input is a full (non-sharded)
    tensor. The style redistributes it to Shard(-1) before the GEMM and
    all-reduces the partial outputs back to Replicate.
    """
    in_features = size
    out_features = size
    batch_size = 4

    # Create reference model (non-parallel)
    ref_model = nn.Linear(in_features, out_features, bias=False).cuda()
    ref_blockwise = Float8BlockwiseLinear.from_float(copy.deepcopy(ref_model))

    # Create parallel model
    parallel_model = nn.Linear(in_features, out_features, bias=False).cuda()
    parallel_model.load_state_dict(ref_model.state_dict())
    parallel_blockwise = Float8BlockwiseLinear.from_float(parallel_model)

    # input_layouts=(Replicate(),) because x is a full tensor on each rank;
    # the style scatters it to Shard(-1) before the FP8 GEMM.
    parallel_blockwise = parallelize_module(
        parallel_blockwise,
        device_mesh,
        Float8BlockwiseRowwiseParallel(input_layouts=(Replicate(),)),
    )

    x = torch.randn(batch_size, in_features, device="cuda", requires_grad=True)
    x_parallel = x.clone().detach().requires_grad_(True)

    # Forward
    out_ref = ref_blockwise(x)
    out_parallel = parallel_blockwise(x_parallel)

    if isinstance(out_parallel, DTensor):
        out_parallel_gathered = out_parallel.redistribute(
            placements=[Replicate()]
        ).to_local()
    else:
        out_parallel_gathered = out_parallel

    sqnr = compute_error(out_ref, out_parallel_gathered)
    assert sqnr >= 25.0, f"Forward SQNR: {sqnr.item()} must be >= 25.0"

    # Backward pass
    out_ref.sum().backward()
    out_parallel.sum().backward()

    if isinstance(x_parallel.grad, DTensor):
        x_parallel_grad = x_parallel.grad.redistribute(
            placements=[Replicate()]
        ).to_local()
    else:
        x_parallel_grad = x_parallel.grad

    sqnr_bwd = compute_error(x.grad, x_parallel_grad)
    assert sqnr_bwd >= 25.0, f"Backward SQNR: {sqnr_bwd.item()} must be >= 25.0"

    print(
        f"✓ RowwiseParallel test passed (fwd SQNR: {sqnr:.2f}, bwd SQNR: {sqnr_bwd:.2f})"
    )


def _test_mixed_parallel_mlp(device_mesh, size=256):
    """Test MLP with mixed colwise + rowwise parallelism.

    fc1: ColwiseParallel  — input Replicate → output Shard(-1)
    fc2: RowwiseParallel  — input Shard(-1)  → output Replicate (all-reduce)
    No extra communication is needed between the two layers.
    """
    in_features = size
    hidden_features = size
    out_features = size
    batch_size = 4

    # Create reference model
    ref_model = SimpleModel(in_features, hidden_features, out_features).cuda()
    ref_model.fc1 = Float8BlockwiseLinear.from_float(ref_model.fc1)
    ref_model.fc2 = Float8BlockwiseLinear.from_float(ref_model.fc2)

    # Create parallel model
    parallel_model = SimpleModel(in_features, hidden_features, out_features).cuda()
    parallel_model.load_state_dict(ref_model.state_dict())
    parallel_model.fc1 = Float8BlockwiseLinear.from_float(parallel_model.fc1)
    parallel_model.fc2 = Float8BlockwiseLinear.from_float(parallel_model.fc2)

    # fc1 colwise, fc2 rowwise — output of fc1 (Shard(-1)) feeds fc2 (expects Shard(-1))
    parallelize_module(
        parallel_model,
        device_mesh,
        {
            "fc1": Float8BlockwiseColwiseParallel(),
            "fc2": Float8BlockwiseRowwiseParallel(),
        },
    )

    x = torch.randn(batch_size, in_features, device="cuda", requires_grad=True)
    x_parallel = x.clone().detach().requires_grad_(True)

    # Forward
    out_ref = ref_model(x)
    out_parallel = parallel_model(x_parallel)

    if isinstance(out_parallel, DTensor):
        out_parallel_gathered = out_parallel.redistribute(
            placements=[Replicate()]
        ).to_local()
    else:
        out_parallel_gathered = out_parallel

    sqnr = compute_error(out_ref, out_parallel_gathered)
    assert sqnr >= 20.0, f"Forward SQNR: {sqnr.item()} must be >= 20.0"

    # Backward pass
    out_ref.sum().backward()
    out_parallel.sum().backward()

    if isinstance(x_parallel.grad, DTensor):
        x_parallel_grad = x_parallel.grad.redistribute(
            placements=[Replicate()]
        ).to_local()
    else:
        x_parallel_grad = x_parallel.grad

    sqnr_bwd = compute_error(x.grad, x_parallel_grad)
    assert sqnr_bwd >= 20.0, f"Backward SQNR: {sqnr_bwd.item()} must be >= 20.0"

    print(
        f"✓ Mixed parallel MLP test passed (fwd SQNR: {sqnr:.2f}, bwd SQNR: {sqnr_bwd:.2f})"
    )


def _test_dtensor_creation_and_redistribution(device_mesh, size=256):
    """Test DTensor creation and redistribution with blockwise linear"""
    in_features = size
    out_features = size
    batch_size = 4
    world_size = device_mesh.size()

    model = nn.Linear(in_features, out_features, bias=False).cuda()
    blockwise = Float8BlockwiseLinear.from_float(model)  # noqa: F841

    x = torch.randn(batch_size, in_features, device="cuda")

    # Create DTensor with Shard(0) placement
    x_dtensor = DTensor.from_local(x, device_mesh, [Shard(0)], run_check=False)
    assert isinstance(x_dtensor, DTensor)
    assert x_dtensor.placements[0] == Shard(0)

    # Redistribute to Replicate
    x_replicated = x_dtensor.redistribute(placements=[Replicate()])
    assert x_replicated.placements[0] == Replicate()

    # After all-gather the batch dimension doubles
    x_local = x_replicated.to_local()
    expected_shape = (batch_size * world_size, in_features)
    assert x_local.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {x_local.shape}"
    )

    print("✓ DTensor creation and redistribution test passed")


def _test_weight_sharding(device_mesh, size=256):
    """Test that weights are properly sharded after parallelize_module"""
    in_features = size
    out_features = size

    model = nn.Linear(in_features, out_features, bias=False).cuda()
    blockwise = Float8BlockwiseLinear.from_float(model)

    blockwise_parallel = parallelize_module(
        blockwise,
        device_mesh,
        Float8BlockwiseColwiseParallel(),
    )

    assert isinstance(blockwise_parallel.weight, DTensor), "Weight should be a DTensor"
    assert blockwise_parallel.weight.placements[0] == Shard(0), (
        "Weight should be sharded on dim 0"
    )

    world_size = device_mesh.size()
    local_weight = blockwise_parallel.weight.to_local()
    expected_out_features = out_features // world_size
    assert local_weight.shape == (expected_out_features, in_features), (
        f"Expected shape ({expected_out_features}, {in_features}), got {local_weight.shape}"
    )

    print("✓ Weight sharding test passed")


if __name__ == "__main__":
    device_mesh = setup_distributed()

    tests = [
        _test_colwise_parallel_forward_backward,
        _test_rowwise_parallel_forward_backward,
        _test_mixed_parallel_mlp,
        _test_dtensor_creation_and_redistribution,
        _test_weight_sharding,
    ]

    print(
        f"Running {len(tests)} tests on rank "
        f"{torch.distributed.get_rank()}/{torch.distributed.get_world_size()}"
    )

    for test in tqdm(
        tests, desc="Running tests", disable=torch.distributed.get_rank() != 0
    ):
        try:
            test(device_mesh)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            raise e

    if torch.distributed.get_rank() == 0:
        print(f"\n✅ All {len(tests)} tests passed!")

    torch.distributed.destroy_process_group()

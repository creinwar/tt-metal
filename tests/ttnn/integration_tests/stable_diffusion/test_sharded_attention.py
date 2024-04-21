# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import pytest
import ttnn

import tt_lib as ttl
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_grayskull,
)
from models.utility_functions import comp_pcc, tt2torch_tensor, torch2tt_tensor, skip_for_wormhole_b0
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_utility_functions import (
    determine_largest_subblock_size,
)


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [4096, 1024])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT16])
def test_time_sharded_attnention_hwb(
    device,
    seq_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
    pytest.skip()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    M = seq_len
    K = 64
    N = seq_len

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )
    block_sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    attn_weights_qkt = torch_query_layer @ torch_key_layer_transposed
    attn_weights_torch_sm = torch.nn.functional.softmax(attn_weights_qkt, dim=-1)
    attn_weights_torch = attn_weights_torch_sm @ torch_value_layer

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    mm_out = torch2tt_tensor(
        torch_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_output_block_shard_spec = [seq_len // 8, seq_len // 8]
    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]

    heads_per_slice = num_heads // num_slices
    for i in range(num_slices):
        q_slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            ttl.tensor.CoreCoord(1, grid_size[0]),
            [M // grid_size[0], K],
            num_slices,
            i,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        k_slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_key_layer_transposed,
            ttl.tensor.CoreCoord(grid_size[1], 1),
            [K, N // grid_size[1]],
            num_slices,
            i,
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=M // (32 * grid_size[0]),
            per_core_N=N // (32 * grid_size[1]),
            transpose_mcast=False,
            fused_activation=None,
        )

        mm_slice = ttl.operations.primary.matmul(
            q_slice,
            k_slice,
            program_config=program_config,
            output_mem_config=block_sharded_mem_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        # mmt = tt2torch_tensor(mm_slice)
        # passed, message = comp_pcc(mmt, attn_weights_qkt[:, i * heads_per_slice : (i + 1) * heads_per_slice, :, :])
        # print(message)
        # assert passed
        k_slice.deallocate()
        q_slice.deallocate()

        height_per_core = seq_len // 64
        output_shard_grid = ttl.tensor.CoreRangeSet(
            {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(7, 7))}
        )
        output_shard_spec = ttl.tensor.ShardSpec(
            output_shard_grid, [height_per_core, seq_len], ttl.tensor.ShardOrientation.ROW_MAJOR, False
        )
        output_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec
        )
        mm_slice = ttl.tensor.reshard(
            mm_slice,
            output_mem_config,
        )
        mm_slice = ttl.tensor.move_sharded(mm_slice)

        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )
        # print(program_config)

        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)
        # mmt = tt2torch_tensor(mm_slice)
        # passed, message = comp_pcc(mmt, attn_weights_torch_sm[:, i * heads_per_slice : (i + 1) * heads_per_slice, :, :])
        # print(message)
        # assert passed

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        v_slice = ttl.tensor.unpad(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), seq_len - 1, 63),
            output_mem_config=dram_interleaved_memory_config,
        )

        mm_slice = ttl.operations.primary.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            output_mem_config=height_sharded_mem_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttl.tensor.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

    mm_out_torch = tt2torch_tensor(mm_out)

    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [4096, 1024])
@pytest.mark.parametrize("num_slices", [16])
@pytest.mark.parametrize("num_cores", [64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
def test_time_sharded_attnention(
    device,
    seq_len,
    num_slices,
    num_cores,
    num_heads,
    data_format,
    function_level_defaults,
):
    pytest.skip()
    compute_grid_size = device.compute_with_storage_grid_size()
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")
    grid_size = (8, 8)

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    mm_out = torch2tt_tensor(
        torch_output,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    tiles_per_shard = math.ceil((((num_heads * seq_len) / num_cores) / num_slices) / 32)
    mm_activations_height_shard_spec = [tiles_per_shard * 32, 2 * 32]
    mm_output_height_shard_spec = [tiles_per_shard * 32, seq_len]
    heads_per_slice = num_heads // num_slices
    for i in range(num_slices):
        slice = ttl.tensor.interleaved_to_sharded_partial(
            reference_query_layer,
            grid_size,
            mm_activations_height_shard_spec,
            num_slices,
            i,
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=2,
            per_core_M=tiles_per_shard,
            per_core_N=seq_len // 32,
            out_subblock_h=1,
            out_subblock_w=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        k_slice = ttl.tensor.unpad(
            reference_key_layer_transposed,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), 63, seq_len - 1),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            slice,
            k_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        k_slice.deallocate()
        slice.deallocate()

        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=mm_output_height_shard_spec[0] // 32,
            block_w=mm_output_height_shard_spec[1] // 32,
        )

        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=seq_len // 32,
            per_core_M=tiles_per_shard,
            per_core_N=2,
            out_subblock_h=1,
            out_subblock_w=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        v_slice = ttl.tensor.unpad(
            reference_value_layer,
            (0, (i * heads_per_slice), 0, 0),
            (0, (i * heads_per_slice) + (heads_per_slice - 1), seq_len - 1, 63),
            output_mem_config=dram_interleaved_memory_config,
        )
        mm_slice = ttl.operations.primary.matmul(
            mm_slice,
            v_slice,
            program_config=program_config,
            output_mem_config=height_sharded_memory_config,
            output_dtype=data_format,
            compute_kernel_config=compute_kernel_config,
        )
        v_slice.deallocate()

        ttl.tensor.sharded_to_interleaved_partial(
            mm_slice,
            mm_out,
            num_slices,
            i,
            dram_interleaved_memory_config,
        )

        mm_slice.deallocate()

    mm_out_torch = tt2torch_tensor(mm_out)

    attn_weights = ttl.tensor.bmm(
        reference_query_layer, reference_key_layer_transposed, output_mem_config=dram_interleaved_memory_config
    )
    attn_weights = ttl.operations.primary.softmax_in_place(attn_weights)
    attn_weights = ttl.tensor.bmm(attn_weights, reference_value_layer, output_mem_config=dram_interleaved_memory_config)

    attn_weights_torch = tt2torch_tensor(attn_weights)
    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [4096, 1024, 256, 64])
@pytest.mark.parametrize("kv_len", [96])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("reshard_for_softmax", [True, False])
def test_cross_attnention(
    device,
    seq_len,
    kv_len,
    num_heads,
    data_format,
    reshard_for_softmax,
    function_level_defaults,
):
    # pytest.skip()
    if seq_len == 64 and reshard_for_softmax:
        pytest.skip()
    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = (8, 2)
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, kv_len]
    value_layer_shape = [1, num_heads, kv_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    q_sharded = ttl.tensor.interleaved_to_sharded(
        reference_query_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=kv_len // 32,
    )
    print(program_config)

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_slice = ttl.operations.primary.matmul(
        q_sharded,
        reference_key_layer_transposed,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    q_sharded.deallocate()

    if reshard_for_softmax:
        if seq_len == 1024:
            mm_slice = ttl.tensor.sharded_to_interleaved(mm_slice, dram_interleaved_memory_config)
            mm_slice = ttl.tensor.interleaved_to_sharded(
                mm_slice,
                (8, 8),
                [height_per_core, kv_len],
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.COL_MAJOR,
            )
        else:
            height_per_core = num_heads * seq_len // 64
            orig_mem_config = mm_slice.memory_config()
            output_shard_grid = ttl.tensor.CoreRangeSet(
                {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(7, 7))}
            )
            output_shard_spec = ttl.tensor.ShardSpec(
                output_shard_grid, [height_per_core, kv_len], ttl.tensor.ShardOrientation.COL_MAJOR, False
            )
            output_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec
            )
            mm_slice = ttl.tensor.reshard(
                mm_slice,
                output_mem_config,
            )
            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                subblock_w=1,
                block_h=32,
                block_w=3,
            )
            mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)
            mm_slice = ttl.tensor.reshard(mm_slice, orig_mem_config)

    else:
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=seq_len // 32,
            block_w=kv_len // 32,
        )
        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

    v_sharded = ttl.tensor.interleaved_to_sharded(
        reference_value_layer,
        grid_size,
        [num_heads * kv_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=kv_len // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=2,
    )
    mm_slice = ttl.operations.primary.matmul(
        mm_slice,
        v_sharded,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    v_sharded.deallocate()

    mm_out_torch = tt2torch_tensor(mm_slice)

    attn_weights_torch = torch_query_layer @ torch_key_layer_transposed
    attn_weights_torch = torch.nn.functional.softmax(attn_weights_torch, dim=-1)
    attn_weights_torch = attn_weights_torch @ torch_value_layer

    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


# Test matmul attention sequence with InterleavedToShardedPartialOp
@skip_for_grayskull()
@pytest.mark.parametrize("seq_len", [1024, 256, 64])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("reshard_for_softmax", [True, False])
def test_attention(
    device,
    seq_len,
    num_heads,
    data_format,
    reshard_for_softmax,
    function_level_defaults,
):
    # pytest.skip()
    if seq_len == 64 and reshard_for_softmax:
        pytest.skip()
    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = (2, 8)
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    query_layer_shape = [1, num_heads, seq_len, 64]
    key_layer_transposed_shape = [1, num_heads, 64, seq_len]
    value_layer_shape = [1, num_heads, seq_len, 64]
    output_shape = [1, num_heads, seq_len, 64]

    torch_query_layer = torch.randn(query_layer_shape).bfloat16().float()
    torch_key_layer_transposed = torch.randn(key_layer_transposed_shape).bfloat16().float()
    torch_value_layer = torch.randn(value_layer_shape).bfloat16().float()
    torch_output = torch.randn(output_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    reference_query_layer = torch2tt_tensor(
        torch_query_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_key_layer_transposed = torch2tt_tensor(
        torch_key_layer_transposed,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    reference_value_layer = torch2tt_tensor(
        torch_value_layer,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    q_sharded = ttl.tensor.interleaved_to_sharded(
        reference_query_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )
    M = num_heads * seq_len
    K = 64
    N = seq_len
    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M // num_cores // 32,
        per_core_N=N // 32,
    )
    print(program_config)

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    mm_slice = ttl.operations.primary.matmul(
        q_sharded,
        reference_key_layer_transposed,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    q_sharded.deallocate()

    if reshard_for_softmax:
        height_per_core = num_heads * seq_len // 64
        orig_mem_config = mm_slice.memory_config()
        if seq_len == 1024:
            mm_slice = ttl.tensor.sharded_to_interleaved(mm_slice, l1_interleaved_memory_config)
            mm_slice = ttl.tensor.interleaved_to_sharded(
                mm_slice,
                (8, 8),
                [height_per_core, seq_len],
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.ROW_MAJOR,
            )
            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                subblock_w=1,
                block_h=height_per_core // 32,
                block_w=seq_len // 32,
            )
            mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)
            mm_slice = ttl.tensor.sharded_to_interleaved(mm_slice, l1_interleaved_memory_config)
            mm_slice = ttl.tensor.interleaved_to_sharded(
                mm_slice,
                (8, 2),
                [num_heads * seq_len // 16, seq_len],
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttl.tensor.ShardOrientation.COL_MAJOR,
            )

        else:
            output_shard_grid = ttl.tensor.CoreRangeSet(
                {ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(7, 7))}
            )
            output_shard_spec = ttl.tensor.ShardSpec(
                output_shard_grid, [height_per_core, seq_len], ttl.tensor.ShardOrientation.COL_MAJOR, False
            )
            output_mem_config = ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, output_shard_spec
            )
            mm_slice = ttl.tensor.reshard(
                mm_slice,
                output_mem_config,
            )
            softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                subblock_w=1,
                block_h=height_per_core // 32,
                block_w=seq_len // 32,
            )
            mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)
            mm_slice = ttl.tensor.reshard(mm_slice, orig_mem_config)
    else:
        softmax_program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid_size,
            subblock_w=1,
            block_h=seq_len // 32,
            block_w=seq_len // 32,
        )
        print(softmax_program_config)
        mm_slice = ttl.operations.primary.softmax_in_place(mm_slice, program_config=softmax_program_config)

    v_sharded = ttl.tensor.interleaved_to_sharded(
        reference_value_layer,
        grid_size,
        [num_heads * seq_len // num_cores, 64],
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    program_config = ttl.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=seq_len // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=num_heads * seq_len // num_cores // 32,
        per_core_N=2,
    )
    print(program_config)
    mm_slice = ttl.operations.primary.matmul(
        mm_slice,
        v_sharded,
        program_config=program_config,
        output_mem_config=height_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    v_sharded.deallocate()

    mm_out_torch = tt2torch_tensor(mm_slice)

    attn_weights_torch = torch_query_layer @ torch_key_layer_transposed
    attn_weights_torch = torch.nn.functional.softmax(attn_weights_torch, dim=-1)
    attn_weights_torch = attn_weights_torch @ torch_value_layer

    passing, output = comp_pcc(mm_out_torch, attn_weights_torch)

    print(output)
    assert passing


@skip_for_grayskull()
@pytest.mark.parametrize("size", [4096, 1024, 256, 64])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT16])
@pytest.mark.parametrize("interleaved_output", [True, False])
def test_qkv(
    device,
    size,
    data_format,
    interleaved_output,
    function_level_defaults,
):
    pytest.skip()
    sizes = {
        4096: [1, 8192, 320, 1536],
        1024: [1, 2048, 640, 2304],
        256: [1, 512, 1280, 3840],
        64: [1, 128, 1280, 3840],
    }
    grid_sizes = {4096: (8, 5), 1024: (8, 5), 256: (8, 8), 64: (4, 8)}
    out_subblock_hs = {4096: 8, 1024: 8, 256: 2, 64: 1}
    B, M, K, N = sizes[size]
    grid_size = grid_sizes[size]
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [1, B, M, K]
    in_1_shape = [1, B, K, N]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    block_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    in_0_sharded = ttl.tensor.interleaved_to_sharded(
        in_0,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    Nt = N // 32
    G = grid_size[1]
    per_core_N = (Nt - 1) // (G - 1) if Nt != 16 else 4
    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // grid_size[1] // 32,
        out_subblock_h=out_subblock_hs[size] if interleaved_output else 1,
        out_subblock_w=1,
        per_core_M=M // grid_size[0] // 32,
        per_core_N=per_core_N,
        fused_activation=None,
        transpose_mcast=True,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm = ttl.operations.primary.matmul(
        in_0_sharded,
        in_1,
        program_config=program_config,
        output_mem_config=dram_interleaved_memory_config if interleaved_output else block_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    # mm = ttl.tensor.bmm(
    #     in_0,
    #     in_1,
    #     l1_interleaved_memory_config,
    #     compute_kernel_config,
    # )

    mm_out_torch = tt2torch_tensor(mm)

    out_torch = in_0_torch @ in_1_torch

    passing, output = comp_pcc(mm_out_torch, out_torch)

    print(output)
    assert passing


@skip_for_grayskull()
@pytest.mark.parametrize("size", [4096, 1024, 256, 64])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
def test_q_and_kv(
    device,
    size,
    data_format,
    function_level_defaults,
):
    # pytest.skip()
    sizes = {4096: [1, 8192, 320, 512], 1024: [1, 2048, 640, 768], 256: [1, 512, 1280, 1280], 64: [1, 128, 1280, 1280]}
    grid_sizes = {4096: (2, 8), 1024: (2, 8), 256: (8, 8), 64: (8, 4)}

    # if size == 4096 and not is_kv:
    #     pytest.skip()

    B, M, K, N = sizes[size]
    grid_size = grid_sizes[size]
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [1, B, M, K]
    in_1_shape = [1, B, K, N]
    in_2_shape = [1, B, 192, K]
    in_3_shape = [1, B, K, 2 * N]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()
    in_2_torch = torch.randn(in_2_shape).bfloat16().float()
    in_3_torch = torch.randn(in_3_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    block_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_2 = torch2tt_tensor(
        in_2_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_3 = torch2tt_tensor(
        in_3_torch,
        device,
        tt_memory_config=dram_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    in_0_sharded = ttl.tensor.interleaved_to_sharded(
        in_0,
        grid_size,
        [M // grid_size[1], K // grid_size[0]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )
    M, K = in_0.shape[-2], in_0.shape[-1]
    N = in_1.shape[-1]
    in0_block_h = M // grid_size[1] // 32
    in0_block_w = K // grid_size[0] // 32
    out_block_h = math.ceil(M / grid_size[1] / 32)
    out_block_w = math.ceil(N / grid_size[0] / 32)
    out_subblock_h, out_subblock_w = determine_largest_subblock_size(out_block_h, out_block_w)
    program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=None,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm_q = ttnn.experimental.operations.primary.matmul(
        in_0,
        in_1,
        program_config=program_config,
        output_mem_config=block_sharded_memory_config,
        output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
        compute_kernel_config=compute_kernel_config,
    )
    in_0_sharded.deallocate()

    M, K, N = in_2.shape[-2], in_2.shape[-1], in_3.shape[-1]
    in0_block_h = M // grid_size[1] // 32
    in0_block_w = K // grid_size[0] // 32
    out_block_h = math.ceil(M / grid_size[1] / 32)
    out_block_w = math.ceil(N / grid_size[0] / 32)
    out_subblock_h, out_subblock_w = determine_largest_subblock_size(out_block_h, out_block_w)
    program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=None,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm_kv = ttnn.experimental.operations.primary.matmul(
        in_2,
        in_3,
        program_config=program_config,
        output_mem_config=block_sharded_memory_config,
        output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
        compute_kernel_config=compute_kernel_config,
    )
    mm_q = ttnn.reshape(mm_q, [2, 1, -1, mm_q.shape[-1]])
    mm_kv = ttnn.reshape(mm_kv, [2, 1, -1, mm_kv.shape[-1]])

    end_core = ttnn.experimental.tensor.CoreCoord(7, 1) if size != 64 else ttnn.experimental.tensor.CoreCoord(3, 1)
    grid_size = (8, 2) if size != 64 else (4, 2)
    output_shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), end_core)}
    )
    output_shard_spec = ttnn.experimental.tensor.ShardSpec(
        output_shard_grid,
        [size, mm_q.shape[-1] // grid_size[0]],
        ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )
    output_mem_config = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.experimental.tensor.BufferType.L1,
        output_shard_spec,
    )
    mm_q = ttnn.experimental.tensor.reshard(
        mm_q,
        output_mem_config,
    )

    output_shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {ttnn.experimental.tensor.CoreRange(ttnn.experimental.tensor.CoreCoord(0, 0), end_core)}
    )
    output_shard_spec = ttnn.experimental.tensor.ShardSpec(
        output_shard_grid,
        [96, mm_kv.shape[-1] // grid_size[0]],
        ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )
    output_mem_config = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.experimental.tensor.BufferType.L1,
        output_shard_spec,
    )
    mm_kv = ttnn.experimental.tensor.reshard(
        mm_kv,
        output_mem_config,
    )

    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1)
    q, k, v = ttl.tensor.create_qkv_heads_from_separate_tensors(
        mm_q,
        mm_kv,
        num_q_heads=8,
        num_kv_heads=8,
        transpose_k_heads=True,
        output_mem_config=out_mem_config,
    )

    mm_out_torch = tt2torch_tensor(mm_q)

    out_torch = in_0_torch @ in_1_torch

    passing, output = comp_pcc(mm_out_torch, out_torch)

    print(output)
    assert passing


@skip_for_grayskull()
@pytest.mark.parametrize("size", [4096, 1024, 256, 64])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("interleaved_output", [True, False])
def test_out(
    device,
    size,
    data_format,
    interleaved_output,
    function_level_defaults,
):
    pytest.skip()
    sizes = {4096: [1, 8192, 512, 320], 1024: [1, 2048, 768, 640], 256: [1, 512, 1280, 1280], 64: [1, 128, 1280, 1280]}
    grid_sizes = {4096: (8, 8), 1024: (8, 4), 256: (8, 5), 64: (4, 8)}
    shard_direction = {
        4096: ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
        1024: ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        256: ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        64: ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
    }
    out_subblock_hs = {256: 8, 64: 4}

    B, M, K, N = sizes[size]
    grid_size = grid_sizes[size]
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [1, B, M, K]
    in_1_shape = [1, B, K, N]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    block_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    hs = shard_direction[size] == ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    if hs:
        in_0_sharded = ttl.tensor.interleaved_to_sharded(
            in_0,
            grid_size,
            [B * M // num_cores, K],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        output_mem_config = height_sharded_memory_config
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // 32 if hs else 1,
            per_core_M=B * M // num_cores // 32 if hs else B * M // 32,
            per_core_N=N // 32 if hs else N // num_cores // 32,
            out_subblock_h=1 if hs else out_subblock_hs[size],
            out_subblock_w=2 if hs else 1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False if hs else True,
        )
    else:
        in_0_sharded = ttl.tensor.interleaved_to_sharded(
            in_0,
            grid_size,
            [B * M // grid_size[0], K // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )
        output_mem_config = block_sharded_memory_config

        Nt = N // 32
        G = grid_size[1]
        per_core_N = (Nt - 1) // (G - 1) if Nt != 16 else 4
        program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // grid_size[1] // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=M // grid_size[0] // 32,
            per_core_N=per_core_N,
            fused_activation=None,
            transpose_mcast=True,
        )

    mm = ttl.operations.primary.matmul(
        in_0_sharded,
        in_1,
        program_config=program_config,
        output_mem_config=l1_interleaved_memory_config if interleaved_output else output_mem_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    mm_out_torch = tt2torch_tensor(mm)

    out_torch = in_0_torch @ in_1_torch

    passing, output = comp_pcc(mm_out_torch, out_torch)

    print(output)
    assert passing


@skip_for_grayskull()
@pytest.mark.parametrize("size", [4096, 1024, 256, 64])
@pytest.mark.parametrize("data_format", [ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("interleaved_output", [True, False])
def test_ff(
    device,
    size,
    data_format,
    interleaved_output,
    function_level_defaults,
):
    pytest.skip()
    sizes = {4096: [1, 8192, 1280, 320], 1024: [1, 2048, 640, 768], 256: [1, 512, 1280, 1280], 64: [1, 128, 1280, 1280]}
    grid_sizes = {4096: (8, 5), 1024: (8, 5), 256: (8, 8), 64: (4, 8)}
    out_subblock_hs = {4096: 8, 1024: 8, 256: 2, 64: 1}

    # if size == 4096 and not is_kv:
    #     pytest.skip()

    B, M, K, N = sizes[size]

    grid_size = grid_sizes[size]
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size[0] * grid_size[1]
    if num_cores > (compute_grid_size.x * compute_grid_size.y):
        pytest.skip(f"Need {num_cores} cores to run this test but core grid is {compute_grid_size}")

    in_0_shape = [1, B, M, K]
    in_1_shape = [1, B, K, N]

    in_0_torch = torch.randn(in_0_shape).bfloat16().float()
    in_1_torch = torch.randn(in_1_shape).bfloat16().float()

    dram_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    l1_interleaved_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    height_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    block_sharded_memory_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttl.tensor.BufferType.L1
    )

    # compare output to regular case
    in_0 = torch2tt_tensor(
        in_0_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )
    in_1 = torch2tt_tensor(
        in_1_torch,
        device,
        tt_memory_config=l1_interleaved_memory_config,
        tt_dtype=data_format,
    )

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    passing = True
    output = None

    in_0_sharded = ttl.tensor.interleaved_to_sharded(
        in_0,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    Nt = N // 32
    G = grid_size[1]
    per_core_N = (Nt - 1) // (G - 1)
    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=K // grid_size[1] // 32,
        out_subblock_h=out_subblock_hs[size] if interleaved_output else 1,
        out_subblock_w=1,
        per_core_M=M // grid_size[0] // 32,
        per_core_N=per_core_N,
        fused_activation=None,
        transpose_mcast=True,
    )
    print(program_config)
    # print(f"Nt: {Nt}, G: {grid_size[1]}, Nt-1: {(Nt-1)}, G-1: {(grid_size[1]-1)} pcn: {(Nt-1)/(grid_size[1]-1)}")
    # print(f"Nt/G: {Nt/grid_size[1]}, Nt/(G-1) = {Nt/(grid_size[1]-1)}")

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    mm = ttl.operations.primary.matmul(
        in_0_sharded,
        in_1,
        program_config=program_config,
        output_mem_config=dram_interleaved_memory_config if interleaved_output else block_sharded_memory_config,
        output_dtype=data_format,
        compute_kernel_config=compute_kernel_config,
    )
    if interleaved_output:
        mm = ttl.tensor.interleaved_to_sharded(
            mm,
            grid_size,
            [mm.shape[-2] // grid_size[0], mm.shape[-1] // grid_size[1]],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    mm_out_torch = tt2torch_tensor(mm)

    out_torch = in_0_torch @ in_1_torch

    passing, output = comp_pcc(mm_out_torch, out_torch)

    print(output)
    assert passing

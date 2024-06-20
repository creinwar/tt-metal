# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Dict, Optional
import torch
import warnings
import math
import ttnn

# import ttl
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import calculate_shard_grid, roundup
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParams,
    find_closest_common_largest_divisor,
    find_closest_largest_divisor,
    find_closest_largest_divisor_with_num_padding,
)
from ttnn.device import (
    is_grayskull,
    is_wormhole_b0,
)
import ttnn.experimental


def determine_largest_subblock_size(block_height, block_width, fp32_accum=False):
    subblocks = [
        (2, 4),
        (4, 2),
        (1, 8),
        (8, 1),
        (1, 7),
        (7, 1),
        (2, 3),
        (3, 2),
        (1, 6),
        (6, 1),
        (1, 5),
        (5, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 3),
        (3, 1),
        (1, 2),
        (2, 1),
        (1, 1),
    ]
    for subblock_height, subblock_width in subblocks:
        if fp32_accum and subblock_height * subblock_width > 4:
            continue
        if block_height % subblock_height == 0 and block_width % subblock_width == 0:
            if subblock_width != block_width and subblock_height != 1:
                continue
            break
    return subblock_height, subblock_width


def _conv_op_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.opimized_conv_new_1", validate_input_tensors=_conv_op_validate_input_tensors)
def opimized_conv_new_1(
    # *,
    input_tensor: ttnn.Tensor,
    weight_tensor: ttnn.Tensor,
    device: ttnn.Device,
    conv_params: list,
    output_channels: int,
    input_height: int,
    input_width: int,
    untilize_out: bool,
    fuse_relu: bool,
    math_fidelity: ttnn.MathFidelity,
    # parallelization_config: ttnn.experimental.Tensor.OptimizedConvParallelizationConfig=None,
    # block_config: ttnn.experimental.tensor.OptimizedConvBlockConfig=None,
    extra_padding_for_32B_alignment: int,
    output_dtype: ttnn.DataType = ttnn.bfloat16,
    # input_tensor_shape: ttnn.TensorShape=None,
    use_shallow_conv_variant: bool = False,
) -> ttnn.Tensor:
    grid_size = device.compute_with_storage_grid_size()
    ncores = input_tensor.shape[3] // 32
    per_core_out_matrix_h_ntiles, per_core_weight_matrix_w_ntiles = (
        2 * input_height * input_width // 32,
        (output_channels // ncores) // 32,
    )
    print(
        f"ncores = {ncores}, per_core_out_matrix_h_ntiles = {per_core_out_matrix_h_ntiles}, per_core_weight_matrix_w_ntiles = {per_core_weight_matrix_w_ntiles}"
    )
    input_tensor_shape = (2, input_height, input_width, input_tensor.shape[3])
    act_block_h, act_block_w = per_core_out_matrix_h_ntiles, (input_tensor_shape[3]) // 32
    out_subblock_h, out_subblock_w = determine_largest_subblock_size(
        per_core_out_matrix_h_ntiles, per_core_weight_matrix_w_ntiles, False
    )
    print("Output Channels ", output_channels)
    # act_block_h, act_block_w, out_subblock_h, out_subblock_w = (
    #     per_core_out_matrix_h_ntiles,
    #     int(input_tensor_shape[3] / 32),
    #     1,
    #     1,
    # )
    opt_conv_parall_conf = ttnn.experimental.tensor.OptimizedConvParallelizationConfig(
        grid_size=grid_size,
        num_cores_nhw=1,
        num_cores_c=ncores,  # todo remove cores variable
        per_core_out_matrix_height_ntiles=per_core_out_matrix_h_ntiles,
        per_core_out_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
    )
    opt_conv_block_conf = ttnn.experimental.tensor.OptimizedConvBlockConfig(
        act_block_h_ntiles=act_block_h,
        act_block_w_ntiles=act_block_w,
        out_subblock_h_ntiles=out_subblock_h,
        out_subblock_w_ntiles=out_subblock_w,
    )
    ttnn._ttnn.operations.conv2d.opt_conv(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        device=device,
        conv_params=conv_params,
        output_channels=output_channels,
        untilize_out=untilize_out,
        fused_relu=fuse_relu,
        math_fidelity=math_fidelity,
        parallelization_config=opt_conv_parall_conf,
        block_config=opt_conv_block_conf,
        extra_padding_for_32B_alignment=False,
        output_dtype=output_dtype,
        output_mem_config=ttnn.get_memory_config(input_tensor),
        input_tensor_shape=input_tensor_shape,
        use_shallow_conv_variant=use_shallow_conv_variant,
    )

    return


__all__ = []

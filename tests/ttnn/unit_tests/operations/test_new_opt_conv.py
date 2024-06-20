from loguru import logger

import torch
import pytest
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_grayskull, is_wormhole_b0
from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    check_with_pcc,
    check_with_pcc_without_tensor_printout,
    update_process_id,
)
from tt_lib.utils import (
    _nearest_y,
)
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import ttnn
import tt_lib
import math
import os


@pytest.mark.parametrize("device_l1_small_size", [16384], indirect=True)
def test_new_conv(device):
    print("OptimizedConvNew1 Python Side")

    # update_process_id()
    batch_size, input_channel, input_height, input_width = 2, 64, 8, 8
    weight_batch, ncores = 64, 2
    filter_height, filter_width = 3, 3
    input_shape = [1, 1, batch_size * input_height * input_width, input_channel]
    weight_shape = [1, 1, weight_batch * filter_height * filter_width, input_channel]
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16) * 2 - 1
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16) * 2 - 1
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    input_2d_height = batch_size * input_height * input_width
    shard_height = batch_size * input_height * input_width
    print("init complted")
    input_2d_width_padded = _nearest_y(input_channel, ncores * 32)
    shard_width = math.ceil(input_2d_width_padded / ncores)
    shard_orientation = ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
    tensor_memory_layout = ttnn.types.TensorMemoryLayout.WIDTH_SHARDED
    # tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED
    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)
    logger.debug(f"shard_memory_layout={in_sharded_mem_config}")
    input_tensor = ttnn.to_memory_config(tt_input, memory_config=in_sharded_mem_config)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ##op computation
    ttnn.opimized_conv_new_1(
        input_tensor,
        tt_weight,
        device,
        [3, 3, 1, 1, 1, 1],
        weight_batch,
        input_height,
        input_width,
        True,
        True,
        ttnn.MathFidelity.HiFi4,
        10,
    )
    # ttnn.opimized_abc([1, 1, 1, 1], 1024, True, True)
    # ttnn.opimized_conv_new_1(tt_input, tt_weight, [1, 1, 1, 1], 1024, True, True)


@pytest.mark.parametrize("device_l1_small_size", [16384], indirect=True)
def test_new_conv2(device):
    print("OptimizedConvNew1 Python Side")

    # update_process_id()
    batch_size, input_channel, input_height, input_width = 2, 128, 8, 8
    output_channels = 256
    filter_height, filter_width = 3, 3
    input_shape = [1, 1, batch_size * input_height * input_width, input_channel]
    weight_shape = [1, 1, input_channel * filter_height * filter_width, output_channels]
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16) * 2 - 1
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16) * 2 - 1
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    tt_weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    input_2d_height = batch_size * input_height * input_width
    shard_height = batch_size * input_height * input_width
    print("init complted")
    ncores = input_channel // 32
    input_2d_width_padded = _nearest_y(input_channel, ncores * 32)
    shard_width = math.ceil(input_2d_width_padded / ncores)
    shard_orientation = ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
    tensor_memory_layout = ttnn.types.TensorMemoryLayout.WIDTH_SHARDED
    # tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED
    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)
    logger.debug(f"shard_memory_layout={in_sharded_mem_config}")
    input_tensor = ttnn.to_memory_config(tt_input, memory_config=in_sharded_mem_config)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ##op computation
    ttnn.opimized_conv_new_1(
        input_tensor,
        tt_weight,
        device,
        [3, 3, 1, 1, 1, 1],
        output_channels,
        input_height,
        input_width,
        True,
        True,
        ttnn.MathFidelity.HiFi4,
        10,
    )

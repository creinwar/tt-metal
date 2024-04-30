# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_mean_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    dim,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.mean(x, dim=dim, keepdim=True)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.mean(x, dim=dim, keepdim=True)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    success, pcc_value = comp_pcc(ref_value, tt_result, 0.99)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(12, 224, 224)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        1,
        19042500,
    ),
    (
        [(2, 12, 32, 128)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        2,
        8721464,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed",
    (test_sweep_args),
)
def test_mean(input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed, device):
    run_mean_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, dim, data_seed, device)

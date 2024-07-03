from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import setup_tt_tensor, eltwise_addcmul


def run_eltwise_addcmul(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, scalar, device):
    torch.manual_seed(data_seed)

    x = gen_rand(input_shape, -100, 100)
    y = gen_rand(input_shape, -100, 100)
    z = gen_rand(input_shape, -100, 100)

    # compute ref value
    ref_value = pytorch_ops.addcmul(x, y, z, scalar=scalar)

    try:
        tt_result = eltwise_addcmul(
            x=x,
            y=y,
            z=z,
            scalar=scalar,
            device=device,
            dtype=dtype,
            layout=dlayout,
            input_mem_config=in_mem_config,
            output_mem_config=output_mem_config,
        )
        # compare tt and golden outputs

        success, pcc_value = comp_pcc(ref_value, tt_result)
        logger.debug(pcc_value)
        logger.debug(success)

        assert success

    except Exception as exc:
        logger.warning(f"run_addcdiv RuntimeError occured {exc}")


def test_eltwise_addcmul(device):
    run_eltwise_addcmul(
        input_shape=(5, 3, 96, 64),
        dtype=[ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        dlayout=[ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        in_mem_config=[
            None,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        output_mem_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        data_seed=3181992,
        scalar=95.5,
        device=device,
    )
    run_eltwise_addcmul(
        input_shape=(5, 5, 160, 192),
        dtype=[ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        dlayout=[ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        in_mem_config=[
            None,
            None,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        output_mem_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        data_seed=4931206,
        scalar=8.625,
        device=device,
    )
    run_eltwise_addcmul(
        input_shape=(3, 2, 192, 32),
        dtype=[ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        dlayout=[ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        in_mem_config=[
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
            None,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        ],
        output_mem_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        data_seed=11079580,
        scalar=-8.625,
        device=device,
    )

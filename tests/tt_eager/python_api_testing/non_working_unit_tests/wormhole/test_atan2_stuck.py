from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import setup_tt_tensor, eltwise_atan2
from models.utility_functions import tt2torch_tensor


def run_eltwise_atan2(input_shape, dtype, dlayout, in_mem_config, output_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = gen_rand(input_shape, -100, 100)
    y = gen_rand(input_shape, -100, 100)

    x_ref = x.detach().clone()
    y_ref = y.detach().clone()

    # compute ref value
    ref_value = pytorch_ops.atan2(x_ref, y_ref)

    tt_result = eltwise_atan2(
        x=x,
        y=y,
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


test_sweep_args = [
    (
        (4, 7, 32, 96),
        [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT16],
        [ttl.tensor.Layout.TILE, ttl.tensor.Layout.TILE],
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        17155532,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_atan2(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    for i in range(0, 2):
        run_eltwise_atan2(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)

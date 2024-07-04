from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import setup_tt_tensor
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand
from tests.ttnn.python_api_testing.sweep_tests.ttnn_ops import setup_ttnn_tensor, ttnn_tensor_to_torch
from models.utility_functions import tt2torch_tensor


def run_eltwise_xlogy(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    x = gen_rand(input_shape, 0.001, 100)
    y = gen_rand(input_shape, 0.001, 100)

    t0 = setup_tt_tensor(
        x,
        device,
        dlayout,
        in_mem_config[0],
        dtype,
    )
    t1 = setup_tt_tensor(
        y,
        device,
        dlayout,
        in_mem_config[1],
        dtype,
    )

    t2 = ttnn.xlogy(t0, t1, memory_config=out_mem_config)

    y = tt2torch_tensor(t2)


test_sweep_args = [
    (
        (4, 7, 32, 96),
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        [
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ],
        (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)),
        17155532,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_xlogy(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_xlogy(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)

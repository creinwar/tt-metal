from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from models.utility_functions import tilize
from python_api_testing.models.utility_functions import (
    pad_activation,
    pad_weight,
    tilize,
    untilize,
    tilize_to_list,
    print_diff_argmax,
    pad_weight,
    is_close,
)
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import torch


def run_tilize_matmul_test(M, K, N):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_shape = [1, 1, M, K]
    b_shape = [1, 1, K, N]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a = ttl.tensor.Tensor(
        A.flatten().tolist(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device,
    )
    a_t = ttl.tensor.tilize(a)
    b_t = ttl.tensor.Tensor(
        tilize_to_list(B),
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    t2 = ttl.tensor.matmul(a_t, b_t)
    assert t2.shape() == [1, 1, M, N]
    tt_host_rm = t2.to(host).data()
    pyt_got_back = torch.Tensor(tt_host_rm).reshape((1, 1, M, N))
    pyt_got_back_rm = untilize(pyt_got_back)

    ref_bmm = torch.matmul(A.reshape(1, M, K), B.reshape(1, K, N))
    ref_bmm = ref_bmm.reshape(1, 1, M, N)
    passing_pcc, output_pcc = comp_pcc(ref_bmm, pyt_got_back_rm, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    ttl.device.CloseDevice(device)
    assert passing_pcc


def test_run_tilize_matmul_test():
    run_tilize_matmul_test(32, 32, 32)

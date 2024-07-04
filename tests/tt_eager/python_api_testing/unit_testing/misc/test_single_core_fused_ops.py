# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn

import tt_lib as ttl
from tt_lib.utils import (
    is_close,
)

from models.utility_functions import comp_pcc, pad_by_zero

shapes = [[1, 1, 32, 32], [1, 1, 32, 128], [1, 2, 128, 128]]


# still have to skip b/c of very low PCC in WH B0
@pytest.mark.parametrize("shape", shapes)
def test_softmax(shape, device):
    torch.manual_seed(1234)
    x = torch.randn(shape).bfloat16().float()
    xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    xtt = ttnn.softmax_in_place(xt)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = torch.nn.functional.softmax(x, dim=-1)

    passing, output = comp_pcc(pt_out, tt_got_back, 0.95752)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("shape", shapes)
def test_layernorm(shape, device):
    torch.manual_seed(1234)
    x = torch.randn(shape).bfloat16().float()
    gamma = torch.randn([shape[-1]]).bfloat16().float()
    beta = torch.randn([shape[-1]]).bfloat16().float()

    xt = ttl.tensor.Tensor(x, ttl.tensor.DataType.BFLOAT16).to(ttl.tensor.Layout.TILE).to(device)
    gammat = pad_by_zero(gamma, device)[0]
    betat = pad_by_zero(beta, device)[0]

    xtt = ttl.tensor.layernorm(xt, 1e-5, gammat, betat)

    tt_got_back = xtt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    pt_out = torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, 1e-5)

    passing, output = comp_pcc(pt_out, tt_got_back, 0.98630)
    logger.info(output)
    assert passing

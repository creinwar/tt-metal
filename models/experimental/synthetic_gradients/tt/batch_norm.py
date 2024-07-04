# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import numpy as np
import tt_lib as ttl
import ttnn
from models.utility_functions import tt2torch, tilize_to_list


def batchnorm1d_inference(weight, bias, running_mean, running_var, epsilon: float, L: int, device):
    gamma = ttl.tensor.Tensor(
        weight,
        [1, 1, 32, L],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    beta = ttl.tensor.Tensor(
        bias,
        [1, 1, 32, L],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    epsilon = ttl.tensor.Tensor(
        [epsilon] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    running_var = ttl.tensor.Tensor(
        running_var,
        [1, 1, 32, L],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )
    running_mean = ttl.tensor.Tensor(
        running_mean,
        [1, 1, 32, L],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )

    BCHW = ttl.tensor.BcastOpDim.HW
    BCADD = ttl.tensor.BcastOpMath.ADD

    def batchnorm1d_inference_(X):
        var_plus_eps = ttl.tensor.bcast(running_var, epsilon, BCADD, BCHW)
        sqrt_var = ttnn.sqrt(var_plus_eps)
        sqrt_inv = ttnn.reciprocal(sqrt_var)
        x_minus_mean = ttnn.sub(X, running_mean)
        x_div_sqrt = ttnn.mul(x_minus_mean, sqrt_inv)
        x_gamma = ttnn.mul(x_div_sqrt, gamma)
        Y = ttnn.add(x_gamma, beta)
        return Y

    return batchnorm1d_inference_

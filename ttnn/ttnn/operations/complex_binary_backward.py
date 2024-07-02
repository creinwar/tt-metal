# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

import sys

import ttnn

import torch

THIS_MODULE = sys.modules[__name__]

__all__ = []


def _golden_function_complex_backward(torch_op, grad_tensor, input_tensor_a, input_tensor_b, *args, **kwargs):
    pyt_y = torch_op(input_tensor_a, input_tensor_b)
    input_tensor_a.retain_grad()
    input_tensor_b.retain_grad()
    pyt_y.backward(gradient=grad_tensor)
    golden_tensor = [input_tensor_a.grad, input_tensor_b.grad]
    return golden_tensor


complex_add_bw = ttnn.register_operation(
    golden_function=lambda grad, a, b, *args, **kwargs: _golden_function_backward(
        torch.atan2, grad, a, b, *args, **kwargs
    )
)(ttnn._ttnn.operations.binary_backward.complex_add_bw)


__all__ = []

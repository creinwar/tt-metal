# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, Optional


import tt_lib as ttl

import ttnn
import torch


def _create_golden_function(torch_function_name):
    torch_function = getattr(torch, torch_function_name)

    def golden_function(input_tensor: ttnn.Tensor, dim: Optional[Union[int, Tuple[int]]] = None, keepdim=False, **_):
        if dim == None:
            return torch_function(input_tensor, keepdim=keepdim)
        else:
            return torch_function(input_tensor, dim=dim, keepdim=keepdim)

    return golden_function


def _create_golden_function_topk():
    def golden_function(input_tensor: ttnn.Tensor, k: int, dim: Optional[int] = None, largest=True, sorted=True, **_):
        return torch.topk(input_tensor, k, dim=dim, largest=largest, sorted=sorted)

    return golden_function


# Generic reductions
ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.mean, golden_function=_create_golden_function("mean"))
ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.sum, golden_function=_create_golden_function("sum"))
ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.max, golden_function=_create_golden_function("max"))
ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.min, golden_function=_create_golden_function("min"))
ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.var, golden_function=_create_golden_function("var"))
ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.std, golden_function=_create_golden_function("std"))

# Special reductions
ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.argmax, golden_function=_create_golden_function("argmax"))

ttnn.attach_golden_function(ttnn._ttnn.operations.reduction.topk, golden_function=_create_golden_function_topk())


__all__ = []

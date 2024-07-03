# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import tt_lib as ttl

import ttnn
import ttnn.decorators


def _preprocess_golden_function_inputs(args, kwargs):
    input_tensor, args, kwargs = ttnn.reflection.pop_argument("input_tensor", args, kwargs)
    padding, args, kwargs = ttnn.reflection.pop_argument("padding", args, kwargs)

    if len(padding) != len(input_tensor.shape):
        raise RuntimeError("ttnn.pad: padding must be the same length as the input tensor rank")

    for start, end in padding:
        if start < 0 or end < 0:
            raise RuntimeError("ttnn.pad: padding must be non-negative")

    pad_start = tuple(start for start, _ in padding)
    *_, pad_start_height, pad_start_width = pad_start
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if pad_start_height % ttnn.TILE_SIZE != 0 or pad_start_width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.pad: padding end must be a multiple of the tile size on height and width for a tensor in tile layout"
            )

    pad_end = tuple(end for _, end in padding)
    *_, pad_end_height, pad_end_width = pad_end
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if pad_end_height % ttnn.TILE_SIZE != 0 or pad_end_width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.pad: padding end must be a multiple of the tile size on height and width for a tensor in tile layout"
            )

    input_tensor = ttnn.to_torch(input_tensor)

    return (input_tensor, padding, *args), kwargs


def _golden_function(input_tensor: ttnn.Tensor, padding, value):
    import torch

    torch_padding = []
    for dimension in reversed(padding):
        torch_padding.append(dimension[0])
        torch_padding.append(dimension[1])
    return torch.nn.functional.pad(input_tensor, pad=torch_padding, mode="constant", value=value)


def _postprocess_golden_function_outputs(output_tensor, args, kwargs):
    output_tensor = ttnn.decorators.default_postprocess_golden_function_outputs(output_tensor, args, kwargs)
    # Padding always turns the intended shape to the shape with tile padding. For simplicity of the operation
    output_tensor = ttnn.reshape(output_tensor, shape=output_tensor.shape.with_tile_padding())
    return output_tensor


pad = ttnn.register_operation(
    golden_function=_golden_function,
    preprocess_golden_function_inputs=_preprocess_golden_function_inputs,
    postprocess_golden_function_outputs=_postprocess_golden_function_outputs,
)(ttnn._ttnn.operations.data_movement.pad)


def _golden_function(input_tensor: ttnn.Tensor, order: Tuple[int, ...], **_):
    if len(input_tensor.shape) != len(order):
        raise RuntimeError(
            "The number of dimensions in the tensor input does not match the length of the desired ordering"
        )

    return input_tensor.permute(order).contiguous().clone()


def _permute_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def permute_golden():
    import torch

    def golden_function(input_tensor: ttnn.Tensor, dims: Tuple[int], **_):
        return torch.permute(input_tensor, dims)

    return golden_function


permute = ttnn.register_operation(golden_function=permute_golden())(ttnn._ttnn.operations.data_movement.permute)


def _golden_function(tensors, dim=0, **_):
    import torch

    return torch.concat(tensors, dim)


def _concat_validate_input_tensors(operation_name, tensors, dim, *args, **kwargs):
    for input_tensor in tensors:
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32),
            layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )


doc = r"""
concat(tensors: List[ttnn.Tensor], dim: int = 0, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

Concats :attr:`tensors` in the given :attr:`dim`.

Args:
    * :attr:`tensors`: the tensors to be concatenated.
    * :attr:`dim`: the concatenating dimension.

Keyword Args:
    * :attr:`memory_config`: the memory configuration to use for the operation

Example::

    >>> tensor = ttnn.concat(ttnn.from_torch(torch.zeros((1, 1, 64, 32), ttnn.from_torch(torch.zeros((1, 1, 64, 32), dim=3)), device)

    >>> tensor1 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> tensor2 = ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16), device=device)
    >>> output = ttnn.concat([tensor1, tensor2], dim=4)
    >>> print(output.shape)
    [1, 1, 32, 64]

"""
concat = ttnn.register_operation(
    name="ttnn.concat",
    validate_input_tensors=_concat_validate_input_tensors,
    golden_function=_golden_function,
    doc=doc,
)(ttnn._ttnn.operations.data_movement.concat)


def _golden_function(tensor, repeats, dim=0, **_):
    import torch

    return torch.repeat_interleave(tensor, repeats, dim=dim)


repeat_interleave = ttnn.register_operation(golden_function=_golden_function)(
    ttnn._ttnn.operations.data_movement.repeat_interleave
)


def _golden_function(tensor, shape, **_):
    return tensor.repeat(shape[0], shape[1], shape[2], shape[3])


repeat = ttnn.register_operation(golden_function=_golden_function)(ttnn._ttnn.operations.data_movement.repeat)


def _golden_function(input_tensor: ttnn.Tensor, scale_factor: Tuple[float, float], **_):
    import torch

    input_tensor = input_tensor.permute(0, 3, 1, 2)
    ret = torch.nn.functional.upsample(input_tensor, scale_factor=scale_factor)
    ret = ret.permute(0, 2, 3, 1)
    return ret


upsample = ttnn.register_operation(
    golden_function=_golden_function,
)(ttnn._ttnn.operations.data_movement.upsample)

__all__ = []

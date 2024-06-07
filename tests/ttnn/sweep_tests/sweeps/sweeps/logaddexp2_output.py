# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1,)],
    "height": [32, 384, 1024],
    "width": [32, 1024, 4096],
    "input_a_dtype": [ttnn.bfloat16],
    "input_b_dtype": [ttnn.bfloat16],
    "input_a_layout": [ttnn.TILE_LAYOUT],
    "input_b_layout": [ttnn.TILE_LAYOUT],
    "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "out_tensor_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}


def run(
    batch_sizes,
    height,
    width,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_b_memory_config,
    input_a_memory_config,
    out_tensor_memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    low = -60
    high = 100

    torch_input_tensor_a = torch_random(input_shape, low, high, dtype=torch.bfloat16)
    torch_input_tensor_b = torch_random(input_shape, low, high, dtype=torch.bfloat16)
    torch_optional_output = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.logaddexp2(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        device=device,
        layout=input_a_layout,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        device=device,
        layout=input_b_layout,
        memory_config=input_b_memory_config,
    )
    output_tensor = ttnn.from_torch(
        torch_optional_output,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=out_tensor_memory_config,
    )

    ttnn.experimental.tensor.logaddexp2(input_tensor_a, input_tensor_b, output_tensor=output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.99)

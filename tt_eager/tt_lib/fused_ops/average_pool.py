# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl

from typing import Union, List
import ttnn


def run_avg_pool_on_device_wrapper(device):
    def average_pool_2d(x, output_mem_config, output_dtype=None):
        out = ttnn.average_pool_2d(x, memory_config=output_mem_config, dtype=output_dtype)
        return out

    return average_pool_2d

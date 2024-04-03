# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.functional_yolov4.tt.ttnn_resblock import TtResBlock

import ttnn
import tt_lib


class TtDownSample3:
    def __init__(
        self,
        parameters,
    ) -> None:
        print("ttnn_parameter", parameters)
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.res = TtResBlock(parameters.res, 8, True)
        self.c4 = parameters.c4
        self.c5 = parameters.c5

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor_c1 = self.c1(input_tensor)
        output_tensor_c2 = self.c2(output_tensor_c1)
        output_tensor = self.c3(output_tensor_c1)

        output_tensor = self.res(device, output_tensor)
        output_tensor = output_tensor.to(device, self.c4.conv.input_sharded_memory_config)

        output_tensor = self.c4(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c2], dim=3)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c5.conv.input_sharded_memory_config)
        output_tensor = self.c5(output_tensor)

        return ttnn.from_device(output_tensor)

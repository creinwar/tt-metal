# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model

import ttnn
import tt_lib


class TtNeck:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.c4 = parameters.c4
        self.c5 = parameters.c5
        self.c6 = parameters.c6
        self.c7 = parameters.c7
        self.c8 = parameters.c8
        self.c9 = parameters.c9
        self.c10 = parameters.c10
        self.p1 = parameters.p1
        self.p2 = parameters.p2
        self.p3 = parameters.p3

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        #######

        #        # 3 CBN blocks
        #        x1 = self.c1(input_tensor)
        #        x1_b = self.b1(x1)
        #        x1_m = self.relu(x1_b)
        #
        #        x2 = self.c2(x1_m)
        #        x2_b = self.b2(x2)
        #        x2_m = self.relu(x2_b)
        #
        #        x3 = self.c3(x2_m)
        #        x3_b = self.b3(x3)
        #        x3_m = self.relu(x3_b)
        #
        #        # maxpools
        #        x4 = self.p1(x3_m)
        #        x5 = self.p2(x3_m)
        #        x6 = self.p3(x3_m)
        #
        #        # concat the outputs of maxpool and x3_m
        #        conc1 = torch.cat([x4, x5, x6, x3_m], dim=1)
        #
        #        # 4 back2back CBRs
        #        # CBR4-1
        #        x7 = self.c4(conc1)
        #        x7_b = self.b4(x7)
        #        x7_m = self.relu(x7_b)
        #
        #        # CBR4-2
        #        x8 = self.c5(x7_m)
        #        x8_b = self.b5(x8)
        #        x8_m = self.relu(x8_b)
        #
        #        # CBR4-3
        #        x9 = self.c6(x8_m)
        #        x9_b = self.b6(x9)
        #        x9_m = self.relu(x9_b)
        #
        #        # CBR4-4
        #        x10 = self.c7(x9_m)
        #        x10_b = self.b7(x10)
        #        x10_m = self.relu(x10_b)
        #
        #        # upsample
        #        u1 = self.u(x10_m)
        #
        #        # Next CBR block to be concatinated with output of u1
        #        # gets the output of downsample4 module which is dimensions: [1, 512, 20, 20] - make a random tensor with that shape for the purpose of running the neck unit test stand-alone
        #        outDownSample4 = torch.rand([1, 512, 20, 20])
        #        # CBR block for conc2
        #        x11 = self.c7(outDownSample4)
        #        x11_b = self.b7(x11)
        #        x11_m = self.relu(x11_b)
        #
        #        # concat CBR output with output from u1
        #        conc2 = torch.cat([u1, x11_m], dim=1)
        #
        #        # 6 back2back CBRs
        #        # CBR6_1
        #        x12 = self.c7(conc2)
        #        x12_b = self.b7(x12)
        #        x12_m = self.relu(x12_b)
        #
        #        # CBR6_2
        #        x13 = self.c8(x12_m)
        #        x13_b = self.b8(x13)
        #        x13_m = self.relu(x13_b)
        #
        #        # CBR6_3
        #        x14 = self.c7(x13_m)
        #        x14_b = self.b7(x14)
        #        x14_m = self.relu(x14_b)
        #
        #        # CBR6_4
        #        x15 = self.c8(x14_m)
        #        x15_b = self.b8(x15)
        #        x15_m = self.relu(x15_b)
        #
        #        # CBR6_5
        #        x16 = self.c7(x15_m)
        #        x16_b = self.b7(x16)
        #        x16_m = self.relu(x16_b)
        #
        #        # CBR6_6
        #        x17 = self.c9(x16_m)
        #        x17_b = self.b9(x17)
        #        x17_m = self.relu(x17_b)
        #
        #        # upsample
        #        u2 = self.u(x17_m)
        #
        #        # CBR block for conc3
        #        outDownSample3 = torch.rand([1, 256, 40, 40])
        #        x18 = self.c9(outDownSample3)
        #        x18_b = self.b9(x18)
        #        x18_m = self.relu(x18_b)
        #
        #        # concat CBR output with output from u2
        #        conc3 = torch.cat([u2, x18_m], dim=1)
        #
        #        # 5 CBR blocks
        #        # CBR5_1
        #        x19 = self.c9(conc3)
        #        x19_b = self.b9(x19)
        #        x19_m = self.relu(x19_b)
        #
        #        # CBR5_2
        #        x20 = self.c10(x19_m)
        #        x20_b = self.b10(x20)
        #        x20_m = self.relu(x20_b)
        #
        #        # CBR5_3
        #        x21 = self.c9(x20_m)
        #        x21_b = self.b9(x21)
        #        x21_m = self.relu(x21_b)
        #
        #        # CBR5_4
        #        x22 = self.c10(x21_m)
        #        x22_b = self.b10(x22)
        #        x22_m = self.relu(x22_b)
        #
        #        # CBR5_5
        #        x23 = self.c9(x22_m)
        #        x23_b = self.b9(x23)
        #        x23_m = self.relu(x23_b)
        #
        #        return x23_m, x9_m, x16_m
        #
        #        #######
        output_tensor = self.c1(input_tensor)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c3(output_tensor)
        output_tensorc3 = output_tensor

        output_tensor = self.p1(input_tensor)
        output_tensorp1 = output_tensor
        output_tensor = self.p2(output_tensor)
        output_tensorp2 = output_tensor
        output_tensor = self.p3(output_tensor)
        output_tensorp3 = output_tensor

        #        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        #        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensorp1, output_tensorp2, output_tensorp3, output_tensorc3], dim=3)

        output_tensor = self.c4(output_tensor)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c6(output_tensor)
        output_tensor_9m = output_tensor
        output_tensor = self.c7(output_tensor)
        output_tensor = self.u(output_tensor)

        # TODO add ttnn tensor here for testing
        #    input_shape = torch_input_tensor.shape
        #    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        #
        #    input_tensor = input_tensor.reshape(
        #        input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]
        #    )

        outDownSample4 = torch.ones([1, 512, 20, 20])
        outDownSample4 = torch.permute(outDownSample4, (0, 2, 3, 1))
        outDownSample4 = outDownSample4.reshape(
            outDownSample4.shape[0], 1, outDownSample4.shape[1] * outDownSample4.shape[2], outDownSample4.shape[3]
        )
        outDownSample4 = ttnn.from_torch(outDownSample4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        outDownSample4 = outDownSample4.to(device, self.c7.conv.input_sharded_memory_config)
        # CBR block for conc2
        outDownSample4_c7 = self.c7(outDownSample4)
        outDownSample4_b7 = self.b7(outDownSample4_c7)
        outDownSample4_r7 = self.relu(outDownSample4_b7)

        #        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        #        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensor, outDownSample4_r7], dim=3)
        output_tensor = self.c7(output_tensor)
        output_tensor = self.c8(output_tensor)
        output_tensor = self.c7(output_tensor)
        output_tensor = self.c8(output_tensor)
        output_tensor = self.c7(output_tensor)
        output_tensor_16m = output_tensor
        output_tensor = self.c9(output_tensor)
        output_tensor = self.u(output_tensor)
        # CBR block for conc3
        # TODO add ttnn random tensor here
        # outDownSample3 = torch.rand([1, 256, 40, 40])
        outDownSample3 = torch.ones([1, 256, 40, 40])
        outDownSample3 = torch.permute(outDownSample3, (0, 2, 3, 1))
        outDownSample3 = outDownSample3.reshape(
            outDownSample3.shape[0], 1, outDownSample3.shape[1] * outDownSample3.shape[2], outDownSample3.shape[3]
        )
        outDownSample3 = ttnn.from_torch(outDownSample3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        outDownSample3 = outDownSample3.to(device, self.c9.conv.input_sharded_memory_config)
        outDownSample3_c9 = self.c9(outDownSample3)
        outDownSample3_b9 = self.b9(outDownSample3_c9)
        outDownSample3_r9 = self.relu(outDownSample3_b9)
        output_tensor = ttnn.concat([output_tensor, outDownSample3_r9], dim=3)
        output_tensor = self.c9(output_tensor)
        output_tensor = self.c10(output_tensor)
        output_tensor = self.c9(output_tensor)
        output_tensor = self.c10(output_tensor)
        output_tensor = self.c9(output_tensor)
        #        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        #        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        #        output_tensor = ttnn.concat([output_tensor, output_tensor_c3], dim=3)

        #        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c8.conv.input_sharded_memory_config)
        #        output_tensor = self.c8(output_tensor)

        return ttnn.from_device(output_tensor), ttnn.from_device(output_tensor_9m), ttnn.from_device(output_tensor_16m)

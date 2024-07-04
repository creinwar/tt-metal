# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2
import ttnn
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_grayskull


def blazeblock(x, in_channel, out_channel, kernel_size, stride, padding, skip_proj, parameters, i, conv_config, device):
    channel_pad = out_channel - in_channel
    if stride == 2:
        if kernel_size == 3:
            h = ttnn.pad(x, (0, 2, 0, 2), value=0)
        else:
            h = ttnn.pad(x, (1, 2, 1, 2), value=0)

        x = ttnn.MaxPool2d(
            input_tensor=x,
            kernel_size=(stride, stride),
            stride=(stride, stride),
            dtype=ttnn.bfloat16,
            device=device,
            batch_size=x.shape[0],
            input_height=x.shape[-2],
            input_width=x.shape[-1],
            reader_patterns_cache={},
        )
    else:
        h = x

    if skip_proj:
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=parameters[i].skip_proj.weight,
            in_channels=in_channel,
            out_channels=out_channel,
            device=device,
            bias_tensor=parameters[i].skip_proj.bias,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=x.shape[0],
            input_height=x.shape[-2],
            input_width=x.shape[-1],
            conv_config=conv_config,
            conv_op_cache={},
            debug=None,
            groups=1,
        )
    elif channel_pad > 0:
        x = ttnn.pad(x, (0, 0, 0, 0), value=0)

    h = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=parameters[i].convs[0].weight,
        in_channels=in_channel,
        out_channels=in_channel,
        device=device,
        bias_tensor=parameters[i].convs[0].bias,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=h.shape[0],
        input_height=h.shape[-2],
        input_width=h.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=in_channel,
    )

    h = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=parameters[i].convs[1].weight,
        in_channels=in_channel,
        out_channels=out_channel,
        device=device,
        bias_tensor=parameters[i].convs[1].bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=h.shape[0],
        input_height=h.shape[-2],
        input_width=h.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    return ttnn.relu(h + x)


def blazepose(x, parameters, device):
    detection2roi_method = "alignment"
    kp1 = 2
    kp2 = 3
    theta0 = 90 * np.pi / 180
    dscale = 1.5
    dy = 0.0
    b = x.shape[0]
    use_shallow_conv_variant = False
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        height_sharding=True,
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=False,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        act_block_h_override=64,
    )

    torch_weight_tensor = torch.randn([48, 3, 5, 5], dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn([1, 1, 1, 48], dtype=torch.bfloat16).float()

    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.float32)  # , layout = ttnn.TILE_LAYOUT

    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.float32)  # , layout = ttnn.TILE_LAYOUT
    print("Shape of tensors :", x.shape, " ", tt_weight_tensor.shape, " ", tt_bias_tensor.shape)
    # Having parameters creates memory issue.Hence debugging this conv2d using random weights created above
    del parameters
    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        # weight_tensor=parameters.backbone1[0].weight,
        weight_tensor=tt_weight_tensor,
        in_channels=3,
        out_channels=48,
        device=device,
        # bias_tensor=parameters.backbone1[0].bias,
        bias_tensor=tt_bias_tensor,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(2, 2),
        batch_size=1,
        input_height=128,
        input_width=128,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=False,
        groups=1,
    )

    x = ttnn.relu(x)
    in_channel = [48, 48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128]
    out_channel = [48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128, 128]

    i = 2
    for i in range(2, 24):
        if i > 1:
            if i == 5 or i == 9 or i == 16:
                x = blazeblock(
                    x,
                    in_channel[i - 2],
                    out_channel[i - 2],
                    5,
                    2,
                    0,
                    True,
                    parameters.backbone1,
                    i,
                    conv_config,
                    device,
                )
            else:
                x = blazeblock(
                    x,
                    in_channel[i - 2],
                    out_channel[i - 2],
                    5,
                    1,
                    2,
                    False,
                    parameters.backbone1,
                    i,
                    conv_config,
                    device,
                )
        i += 1

    i = 0

    for i in range(6):
        if i == 0:
            h = blazeblock(x, 128, 256, 5, 2, 0, True, parameters.backbone2, i, conv_config, device)
        else:
            h = blazeblock(h, 256, 256, 5, 1, 2, False, parameters.backbone2, i, conv_config, device)
        i += 1

    c1 = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.classifier_8.weight,
        in_channels=128,
        out_channels=2,
        device=device,
        bias_tensor=parameters.classifier_8.bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=x.shape[0],
        input_height=x.shape[-2],
        input_width=x.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    c1 = ttnn.permute(c1, (0, 2, 3, 1))
    c1 = ttnn.reshape(c1, (b, -1, 1))

    c2 = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=parameters.classifier_16.weight,
        in_channels=256,
        out_channels=6,
        device=device,
        bias_tensor=parameters.classifier_16.bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=h.shape[0],
        input_height=h.shape[-2],
        input_width=h.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    c2 = ttnn.permute(c2, (0, 2, 3, 1))
    c2 = ttnn.reshape(c2, (b, -1, 1))

    c = ttnn.concat([c1, c2], dim=1)

    r1 = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.regressor_8.weight,
        in_channels=128,
        out_channels=24,
        device=device,
        bias_tensor=parameters.regressor_8.bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=x.shape[0],
        input_height=x.shape[-2],
        input_width=x.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    r1 = ttnn.permute(r1, (0, 2, 3, 1))
    r1 = ttnn.reshape(r1, (b, -1, 12))

    r2 = ttnn.conv2d(
        input_tensor=h,
        weight_tensor=parameters.regressor_16.weight,
        in_channels=256,
        out_channels=72,
        device=device,
        bias_tensor=parameters.regressor_16.bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=h.shape[0],
        input_height=h.shape[-2],
        input_width=h.shape[-1],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    r2 = ttnn.permute(r2, (0, 2, 3, 1))
    r2 = ttnn.reshape(r2, (b, -1, 12))

    r = ttnn.concat([r1, r2], dim=1)
    return [r, c]

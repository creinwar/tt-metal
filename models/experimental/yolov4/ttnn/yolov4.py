from typing import Any
import ttnn
import torch
import pytest

ttnn.enable_fast_runtime_mode = False
ttnn.enable_logging = True
ttnn.report_name = "yolo_fail"
ttnn.enable_graph_report = False
ttnn.enable_detailed_buffer_report = True
ttnn.enable_detailed_tensor_report = True
ttnn.enable_comparison_mode = False


def fold_bn_to_conv_weights_bias(model, path):
    # Note: this function is not used, however I am keeping it for reference

    bn_weight = model[path + ".conv.1.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = model[path + ".conv.1.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = model[path + ".conv.0.weight"]
    weight = (weight / torch.sqrt(bn_running_var)) * bn_weight

    bn_running_mean = model[path + ".conv.1.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = model[path + ".conv.1.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var)) + bn_bias

    bias = bias.reshape(1, 1, 1, -1)
    print(path, weight.shape, bias.shape)
    return (
        ttnn.from_torch(
            weight,
        ),
        ttnn.from_torch(bias),
    )


class YOLOv4:
    def __init__(self, path) -> None:
        self.torch_model = torch.load(path)
        self.torch_keys = self.torch_model.keys()
        self.down1 = Down1(self)
        self.downs = [self.down1]

    def __call__(self, device, input_tensor):
        output = self.down1(device, input_tensor)
        return output

    def __str__(self) -> str:
        this_str = ""
        for down in self.downs:
            this_str += str(down)
            this_str += " \n"
        return this_str


class Down1:
    def __init__(self, model) -> None:
        torch_model = model.torch_model
        self.conv1 = Conv(torch_model, "down1.conv1", [1, 320, 320, 4], (1, 1, 1, 1), 64)
        self.conv2 = Conv(torch_model, "down1.conv2", [1, 320, 320, 32], (2, 2, 1, 1), 32)
        self.conv3 = Conv(torch_model, "down1.conv3", [1, 160, 160, 64], (1, 1, 1, 1), None)
        self.conv4 = Conv(torch_model, "down1.conv4", [1, 160, 160, 64], (1, 1, 1, 1), None)
        self.conv5 = Conv(torch_model, "down1.conv5", [1, 160, 160, 32], (1, 1, 1, 1), None)
        self.conv6 = Conv(torch_model, "down1.conv6", [1, 160, 160, 64], (1, 1, 1, 1), None)
        self.conv7 = Conv(torch_model, "down1.conv7", [1, 160, 160, 64], (1, 1, 1, 1), None)
        self.conv8 = Conv(torch_model, "down1.conv8", [1, 160, 160, 128], (1, 1, 1, 1), None)
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]

    def __call__(self, device, input_tensor):
        # output_tensor = self.conv1(device,input_tensor)
        output_tensor = input_tensor
        output_tensor_split = self.conv2(device, output_tensor)
        # output_tensor_left = self.conv3(device,output_tensor_split)

        # output_tensor = self.conv4(device,output_tensor_split)
        # output_tensor = self.conv5(device,output_tensor)
        # output_tensor = self.conv6(device,output_tensor)
        # output_tensor = self.conv7(device,output_tensor)
        # output_tensor = self.conv8(device,output_tensor)
        return output_tensor

    def __str__(self) -> str:
        this_str = ""
        for conv in self.convs:
            this_str += str(conv)
            this_str += " \n"
        return this_str


class Conv:
    def __init__(self, model, path, input_params, conv_params, act_block_h=None) -> None:
        self.weights, self.bias = fold_bn_to_conv_weights_bias(model, path)
        self.input_params = input_params
        self.kernel_size = (self.weights.shape[1], self.weights.shape[2])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation="",
            height_sharding=True,
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            input_channels_alignment=16 if self.input_params[3] < 16 else 32,
            transpose_shards=False,
            reshard_if_not_optimal=True,
            deallocate_activation=True,
            reallocate_halo_output=True,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        print("act_block_h", conv_config.act_block_h_override)

        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
        )
        return output_tensor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolo(device):
    yolov4 = YOLOv4("/localdev/smanoj/models/yolov4.pth")
    print(yolov4)
    x = ttnn.from_torch(torch.randn((1, 320, 320, 32), dtype=torch.bfloat16).float(), dtype=ttnn.bfloat16)

    result = yolov4(device, x)

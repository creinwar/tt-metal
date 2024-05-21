# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from ttnn.model_preprocessing import preprocess_model
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


from models.experimental.functional_yolox_m.reference.yolox_m import YOLOX
from models.experimental.functional_yolox_m.tt.ttnn_yolox import TtYOLOX
import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_cspdarknet as cspdarknet
import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_yolopafpn as fpn
import tests.ttnn.integration_tests.yolox_m.custom_preprocessor_yolohead as head


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters["backbone"] = {}
        if isinstance(model, YOLOX):
            parameters["backbone"]["backbone"] = cspdarknet.custom_preprocessor(
                device, model.backbone.backbone, name, ttnn_module_args["backbone"]["backbone"]
            )
            parameters["backbone"].update(
                fpn.custom_preprocessor(device, model.backbone, name, ttnn_module_args["backbone"])
            )
            parameters["head"] = head.custom_preprocessor(device, model.head, name, ttnn_module_args["head"])
        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@skip_for_wormhole_b0()
def test_yolox_model(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolox")
    if model_path == "models":
        state_dict = torch.load("tests/ttnn/integration_tests/yolox_m/yolox_m.pth", map_location="cpu")
    else:
        weights_pth = str(model_path / "yolox_m.pth")
        state_dict = torch.load(weights_pth)

    state_dict = state_dict["model"]
    ds_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(("backbone", "head")))}
    torch_model = YOLOX()
    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 3, 640, 640)
    torch_output_tensors = torch_model(torch_input_tensor)

    torch_output_tensor0 = torch_output_tensors[0]
    torch_output_tensor1 = torch_output_tensors[1]
    torch_output_tensor2 = torch_output_tensors[2]
    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )
    ttnn_model = TtYOLOX(device, parameters)

    # Tensor Preprocessing
    x = torch_input_tensor
    patch_top_left = x[..., ::2, ::2]
    patch_top_right = x[..., ::2, 1::2]
    patch_bot_left = x[..., 1::2, ::2]
    patch_bot_right = x[..., 1::2, 1::2]

    input_tensor0 = torch.permute(patch_top_left, (0, 2, 3, 1))
    input_tensor0 = input_tensor0.reshape(
        input_tensor0.shape[0], 1, input_tensor0.shape[1] * input_tensor0.shape[2], input_tensor0.shape[3]
    )
    input_tensor0 = ttnn.from_torch(input_tensor0, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    input_tensor1 = torch.permute(patch_top_right, (0, 2, 3, 1))
    input_tensor1 = input_tensor1.reshape(
        input_tensor1.shape[0], 1, input_tensor1.shape[1] * input_tensor1.shape[2], input_tensor1.shape[3]
    )
    input_tensor1 = ttnn.from_torch(input_tensor1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    input_tensor2 = torch.permute(patch_bot_left, (0, 2, 3, 1))
    input_tensor2 = input_tensor2.reshape(
        input_tensor2.shape[0], 1, input_tensor2.shape[1] * input_tensor2.shape[2], input_tensor2.shape[3]
    )
    input_tensor2 = ttnn.from_torch(input_tensor2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    input_tensor3 = torch.permute(patch_bot_right, (0, 2, 3, 1))
    input_tensor3 = input_tensor3.reshape(
        input_tensor3.shape[0], 1, input_tensor3.shape[1] * input_tensor3.shape[2], input_tensor3.shape[3]
    )
    input_tensor3 = ttnn.from_torch(input_tensor3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    input_tensor = [input_tensor0, input_tensor1, input_tensor2, input_tensor3]
    output_tensors = ttnn_model(device, input_tensor)

    # Tensor Postprocessing
    output_tensor0 = output_tensors[0]
    output_tensor1 = output_tensors[1]
    output_tensor2 = output_tensors[2]

    output_tensor0 = ttnn.to_torch(output_tensor0)
    output_tensor0 = output_tensor0.reshape(1, 80, 80, 85)
    output_tensor0 = torch.permute(output_tensor0, (0, 3, 1, 2))
    output_tensor0 = output_tensor0.to(torch_input_tensor.dtype)

    output_tensor1 = ttnn.to_torch(output_tensor1)
    output_tensor1 = output_tensor1.reshape(1, 40, 40, 85)
    output_tensor1 = torch.permute(output_tensor1, (0, 3, 1, 2))
    output_tensor1 = output_tensor1.to(torch_input_tensor.dtype)

    output_tensor2 = ttnn.to_torch(output_tensor2)
    output_tensor2 = output_tensor2.reshape(1, 20, 20, 85)
    output_tensor2 = torch.permute(output_tensor2, (0, 3, 1, 2))
    output_tensor2 = output_tensor2.to(torch_input_tensor.dtype)

    assert_with_pcc(torch_output_tensor0, output_tensor0, pcc=0.80)  # PCC = 0.8041133430120856
    assert_with_pcc(torch_output_tensor1, output_tensor1, pcc=0.86)  # PCC = 0.8681695629677733
    assert_with_pcc(torch_output_tensor2, output_tensor2, pcc=0.91)  # PCC = 0.9175503378141336

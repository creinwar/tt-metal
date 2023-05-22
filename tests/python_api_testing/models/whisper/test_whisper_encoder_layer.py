import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import torch.nn as nn
import numpy as np

import random
from typing import Optional, Tuple, Union
from loguru import logger

from transformers import WhisperModel, WhisperConfig, WhisperForAudioClassification

from python_api_testing.models.whisper.whisper_common import (
    torch2tt_tensor,
    tt2torch_tensor,
    create_padded_tensor,
    create_unpadded_tensor,
)
from python_api_testing.models.whisper.whisper_encoder_layer import (
    TtWhisperEncoderLayer,
)

import tt_lib

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def run_whisper_encoder_layer(layer, device, for_audio_classification=False):
    if for_audio_classification:
        model = WhisperForAudioClassification.from_pretrained(
            "sanchit-gandhi/whisper-medium-fleurs-lang-id"
        )
    else:
        model = WhisperModel.from_pretrained("openai/whisper-tiny.en")

    state_dict = model.state_dict()
    configuration = model.config

    embed_dim = configuration.d_model
    encoder_ffn_dim = configuration.encoder_ffn_dim
    num_heads = configuration.encoder_attention_heads

    """ Huggingface Whisper Encoder Layer """
    ENCODER_IND = layer
    pytorch_model = model.encoder.layers[ENCODER_IND]
    pytorch_model.eval()
    base_address = f"encoder.layers.{ENCODER_IND}"

    hidden_state_input_tensor = torch.rand(1, 1500, embed_dim)
    attention_mask_input_tensor = None
    ttm_tensor_attention_mask = None
    layer_head_mask_input_tensor = None

    # Pad input tensor for tt
    output_tensor_shape = list(hidden_state_input_tensor.size())
    output_tensor_shape.insert(0, 1)
    output_tensor_shape[-2] = 1504
    ttm_tensor_hidden_state = create_padded_tensor(
        list(hidden_state_input_tensor.size()),
        hidden_state_input_tensor,
        output_tensor_shape,
        pad_value=0.0,
        device=device,
    )

    with torch.no_grad():
        pytorch_output = pytorch_model(
            hidden_state_input_tensor,
            attention_mask_input_tensor,
            layer_head_mask_input_tensor,
        )

    """ tt_lib Whisper Encoder Layer """

    # layer_head_mask_input_tensor has size [6] and is equal to number of encoder_attention_heads
    # Stays torch tensor and then is used in attention mechanism approprietly for now
    # Because can't convert 1d tensor of size [6] to tt_lib

    tt_whisper_encoder_layer = TtWhisperEncoderLayer(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        config=configuration,
    )
    with torch.no_grad():
        ttm_output = tt_whisper_encoder_layer(
            ttm_tensor_hidden_state,
            ttm_tensor_attention_mask,
            layer_head_mask_input_tensor,
            True,
        )

    # Returns Tuple of size 2
    # Attention weights Not used
    # Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
    # but it is not used
    # First check: attention output

    # Unpad output tensor
    input_tensors_shape = ttm_output[0].shape()
    print(input_tensors_shape)
    input_tensors_shape[-2] = 1500
    ttm_output_to_torch_0 = create_unpadded_tensor(ttm_output[0], input_tensors_shape)
    ttm_output_to_torch_0 = torch.Tensor(ttm_output_to_torch_0.data()).reshape(
        *ttm_output_to_torch_0.shape()
    )
    ttm_output_to_torch_0 = ttm_output_to_torch_0.squeeze(0)

    does_pass, pcc_message = comp_pcc(pytorch_output[0], ttm_output_to_torch_0, 0.98)

    print(comp_allclose(pytorch_output[0], ttm_output_to_torch_0))
    print(pcc_message)

    if does_pass:
        logger.info("Encoder layer Passed!")
    else:
        logger.warning("Encoder layer Failed!")

    assert does_pass


def test_WhipserEncoderLayer_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_encoder_layer(layer=0, device=device)
    tt_lib.device.CloseDevice(device)


def test_WhisperEncoderLayerForAudioClassification_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_encoder_layer(layer=0, device=device, for_audio_classification=True)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_WhipserEncoderLayer_inference()
    test_WhisperEncoderLayerForAudioClassification_inference()

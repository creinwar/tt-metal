# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
from loguru import logger
import torch
from torch import nn
import pytest

from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from typing import Optional, Tuple
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import TrOCRConfig, VisionEncoderDecoderModel
from models.experimental.functional_trocr.reference.functional_torch_trocr import (
    TrOCRForCausalLM,
)

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.reference import torch_functional_vit

import pytest
import torch
import transformers

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_vit.reference import torch_functional_vit


def encoder_decoder(
    encoder_name,
    decoder_name,
    pixel_values: Optional[torch.FloatTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    # return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple[torch.FloatTensor], torch.FloatTensor]:
    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
    kwargs_decoder = {
        argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }

    encoder = transformers.ViTModel.from_pretrained(encoder_name).to(torch.bfloat16)
    if encoder_outputs is None:
        parameters = preprocess_model_parameters(
            initialize_model=lambda: encoder,
            convert_to_ttnn=lambda *_: False,
            custom_preprocessor=torch_functional_vit.custom_preprocessor,
        )
        encoder_state_dict = encoder.state_dict()
        torch_cls_token = torch.nn.Parameter(encoder_state_dict["embeddings.cls_token"])
        torch_position_embeddings = torch.nn.Parameter(encoder_state_dict["embeddings.position_embeddings"])
        seq_out, encoder_output = torch_functional_vit.vit(
            encoder.config,
            pixel_values,
            torch_position_embeddings,
            torch_cls_token,
            attention_mask=None,
            parameters=parameters,
        )

    decoder = VisionEncoderDecoderModel.from_pretrained(decoder_name)
    decoder_config = decoder.decoder.config
    decoder = decoder.decoder
    parameters = preprocess_model_parameters(
        initialize_model=lambda: decoder,
        convert_to_ttnn=lambda *_: False,
    )
    encoder_hidden_states = encoder_outputs[0]  # Need to return hidden state also from encoder
    encoder_hidden_states = encoder_hidden_states.float()
    decoder_output = TrOCRForCausalLM(
        config=decoder_config,
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        parameters=parameters,
    )

    return decoder_output
    passing, pcc_message = comp_pcc(torch_output, decoder_output, 0.99)
    logger.info(f"PCC: {pcc_message}")

    return decoder_output

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
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import inspect
import torch
import transformers
from models.utility_functions import torch_random
from models.experimental.functional_trocr.trocr_utils import generate


@pytest.mark.parametrize("decoder_name", ["microsoft/trocr-base-handwritten"])
@pytest.mark.parametrize("encoder_name", ["google/vit-base-patch16-224"])
@pytest.mark.parametrize("input_shape", [[1, 3, 224, 224]])
def test_model(encoder_name, decoder_name, input_shape):
    torch_pixel_values = torch_random(input_shape, -1, 1, dtype=torch.bfloat16)
    model = VisionEncoderDecoderModel.from_pretrained(decoder_name)
    generated_ids = generate(model.config, model.generation_config, encoder_name, decoder_name, torch_pixel_values)

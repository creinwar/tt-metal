# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from PIL import Image

import requests
from loguru import logger
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_trocr.ttnn_trocr_attention import trocr_attention

from models.experimental.functional_trocr.ttnn_trocr_generation_utils import GenerationMixin


def test_demo(device):
    with torch.no_grad():
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        config = model.decoder.config

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.eval()

        iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")

        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        # iam_ocr_sample_input = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values

        torch_generated_ids = model.generate(pixel_values, device)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model.decoder,
            device=device,
        )

        generationmixin = GenerationMixin(model=model, device=device, config=config, parameters=parameters)

        ttnn_output = generationmixin.generate(pixel_values)

        torch_generated_text = processor.batch_decode(torch_generated_ids, skip_special_tokens=True)[0]

        ttnn_generated_text = processor.batch_decode(ttnn_output, skip_special_tokens=True)[0]

        logger.info(f"Torch output: {torch_generated_text}")
        logger.info(f"Ttnn output: {ttnn_generated_text}")

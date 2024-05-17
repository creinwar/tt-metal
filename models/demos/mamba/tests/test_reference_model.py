# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from typing import Optional
from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.reference.prefill_model import Mamba


def generate_through_selective_scan(
    model, tokenizer, prompt: str, n_tokens_to_gen: int = 30, sample: bool = False, top_k: Optional[int] = None
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, n_tokens_to_gen, sample=sample, top_k=top_k)
    return [tokenizer.decode(out.tolist()) for out in output][0]


def generate_through_decode(model, tokenizer, prompt: str, n_tokens_to_gen: int = 51):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids, n_tokens_to_gen)
    return tokenizer.batch_decode(output)[0]


@pytest.mark.parametrize(
    "model_version, batch, genlen",
    (
        # ("state-spaces/mamba-130m", 1, 32),
        ("state-spaces/mamba-370m", 1, 32),
    ),
)
def test_cpu_reference_model_decode_vs_selective_scan(
    model_version: MambaPretrainedModelName,
    batch: int,
    genlen: int,
):
    prompt = "Mamba is the"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    prefill_model = Mamba.from_pretrained(model_version)
    decode_model = MambaDecode.from_pretrained(model_version)

    prefill_output = generate_through_selective_scan(prefill_model, tokenizer, prompt, genlen)
    decode_output = generate_through_decode(decode_model, tokenizer, prompt, genlen)
    print(f"selective_scan_output: {prefill_output}")
    print(f"decode_output: {decode_output}")
    assert prefill_output == decode_output, "Model outputs should match"

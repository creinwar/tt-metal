# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.mistral7b.tt.mistral_common_ttnn import (
    precompute_freqs,
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
)
from models.demos.mistral7b.tt.mistral_decoder_ttnn import TtTransformerBlock
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mistral7b.reference.model import TransformerBlock
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
@pytest.mark.parametrize(
    "iterations",
    ((3),),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mistral_decoder_inference(pcc, model_config, model_location_generator, device, iterations):
    ttnn.enable_program_cache()
    dtype_str, mem_config_str = model_config.split("-")
    if dtype_str == "BFLOAT16":
        dtype = ttnn.bfloat16
    elif dtype_str == "BFLOAT8":
        dtype = ttnn.bfloat8_b
    else:
        raise ValueError(f"Unknown dtype {dtype_str}")

    model_args = TtModelArgs()
    state_dict = torch.load(model_args.consolidated_weights_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}

    model_args.max_batch_size = 32

    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    # Helper function supports multiple devices but we only use one
    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        [device], model_args.head_dim, model_args.max_seq_len * 2, 10000, dtype
    )
    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
        tt_cos_cached=tt_cos_cached,
        tt_sin_cached=tt_sin_cached,
    )

    generation_start_pos = 0
    generation_length = iterations
    all_tests_pass = True

    seqlen = 1
    batch = 32

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        print(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        start_pos = generation_start_pos + i

        decode_input, start_pos, attn_mask, current_pos, rot_mat = prepare_inputs_ttnn(
            tt_decode_input,
            start_pos,
            model_args.dim,
            model_args.head_dim,
            model_args.sliding_window,
            model_args.max_seq_len,
            tt_model.device,
        )
        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask, rot_mat)
        # tt_output = tt_model(tt_input, bcast_freq_xq, bcast_freq_xk, tt_position, mask, seqlen)

        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])

        # Reference model
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions, mask=None)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Decoder Block Passed!")
        else:
            logger.warning("Mistral Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"

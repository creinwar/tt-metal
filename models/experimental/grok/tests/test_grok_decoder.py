# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from models.experimental.grok.tt.grok_common import prepare_inputs_ttnn, prepare_rotation_mat_ttnn
from models.experimental.grok.tt.grok_decoder import TtTransformerBlock
from models.experimental.grok.reference.model import DecoderLayer
from models.experimental.grok.tt.model_config import TtModelArgs
from models.utility_functions import comp_pcc, comp_allclose
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor


def test_grok_decoder_inference(t3k_device_mesh, use_program_cache, reset_seeds):
    """
    b: batch
    s: sequence length
    h: hidden size
    """
    pcc = 0.985  # FIXME: debug why this is not .99
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    key_start = "model.layers.0."
    partial_state_dict = {k[len(key_start) :]: v for k, v in state_dict.items() if (k.startswith(key_start))}
    reference_model = DecoderLayer(
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        num_heads=model_args.num_attention_heads,
        num_key_value_heads=model_args.num_key_value_heads,
        num_experts=model_args.num_experts,
        top_k=model_args.num_experts_per_tok,
        max_position_embeddings=model_args.max_position_embeddings,
        attn_output_multiplier=model_args.attn_output_multiplier,
        max_attn_val=model_args.max_attn_value,
        rms_norm_eps=model_args.rms_norm_eps,
    )
    reference_model.load_state_dict(partial_state_dict)

    # Initialize TT model
    tt_model = TtTransformerBlock(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
    )

    rot_mat = prepare_rotation_mat_ttnn(
        model_args.head_dim,
        model_args.max_seq_len,
        tt_model.device_mesh,
    )

    ref_past_key_value = None
    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    seqlen = 1
    batch = 32

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 6144)
        pt_decode_input_bsh = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        current_pos = generation_start_pos + i

        decode_input_b1sh, attn_mask = prepare_inputs_ttnn(
            pt_decode_input_bsh,
            model_args.dim,
            current_pos,
            tt_model.device_mesh,
        )
        # Run TT model
        tt_out_b1sh = tt_model(decode_input_b1sh, current_pos, attn_mask, rot_mat)
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del decode_input_b1sh, attn_mask
        tt_output_torch_b1h = (
            ttnn.to_torch(tt_out_b1sh, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch, 1, -1)
        )

        # Reference model
        ref_output_bsh, ref_past_key_value = reference_model(
            pt_decode_input_bsh,
            past_key_value=ref_past_key_value,
            use_cache=True,
        )

        passing, pcc_message = comp_pcc(ref_output_bsh, tt_output_torch_b1h, pcc)

        logger.info(comp_allclose(ref_output_bsh, tt_output_torch_b1h))
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

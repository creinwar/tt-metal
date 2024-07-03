# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn_prefill,
    get_prefill_rot_mat,
    get_rot_transformation_mat,
    set_model_args,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_decoder import TtTransformerBlock
from models.demos.t3000.mixtral8x7b.reference.model import TransformerBlock, precompute_freqs_cis, RMSNorm
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_pcc, comp_allclose
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor


@pytest.mark.parametrize(
    "seq_len",
    (128, 1024, 2048, 4096, 8192, 8192 * 2, 8192 * 4),
)
def test_mixtral_decoder_inference(t3k_device_mesh, use_program_cache, reset_seeds, seq_len):
    """
    b: batch
    s: sequence length
    h: hidden size
    """
    pcc = 0.99
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args = set_model_args(model_args, seq_len)
    batch = 1
    state_dict = model_args.load_state_dict()
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.1."))}
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, t3k_device_mesh, seq_len=seq_len)
    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )

    # Initialize TT model
    tt_model = TtTransformerBlock(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
    )

    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True
    rms_state_dict = {k[18:]: v for k, v in state_dict.items() if (k.startswith("layers.0.ffn_norm."))}
    rms = RMSNorm(dim=model_args.dim)
    rms.load_state_dict(rms_state_dict)
    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input_bsh = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        start_pos = generation_start_pos + i
        current_pos = start_pos % model_args.sliding_window

        decode_input_b1sh, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            pt_decode_input_bsh,
            tt_model.device_mesh,
        )
        # Run TT model
        tt_out_b1sh = tt_model(
            decode_input_b1sh,
            start_pos,
            current_pos,
            attn_mask,
            rot_mats,
            transformation_mats,
            user_id=0,
            mode="prefill",
        )

        tt_output_torch_b1h = (
            ttnn.to_torch(tt_out_b1sh, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .squeeze(1)
            .view(batch, seq_len, -1)
        )

        # Reference model
        positions = torch.LongTensor(range(seq_len))
        freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
        ref_output_bsh = reference_model(pt_decode_input_bsh, freqs_cis_i, positions, mask=attn_mask_torch)
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

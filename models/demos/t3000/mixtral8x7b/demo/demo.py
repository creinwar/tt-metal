# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import json
import tt_lib as ttl
import pytest
from loguru import logger
from time import time
import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn,
    prepare_rotation_mat_ttnn,
    sample,
    cache_attention,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.mixtral_embedding import TtMixtralEmbedding
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.utility_functions import get_devices_for_t3000


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs(input_prompts, tokenizer, model_args, dtype, instruct, devices):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    # Pad the inputs to the max length prompt
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.int32)

    logger.info(f"# of users: {len(encoded_prompts)}")
    for i, encoded in enumerate(encoded_prompts):
        # Right padding
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)

    input_mask_bool = input_tokens != tokenizer.pad_id
    input_mask = input_mask_bool.int()  # from_torch doesn't support bool type

    # convert to ttnn tensor
    # Encoded input tokens need to be uint32 for embedding. Otherwise the dtype conversion to bfloat16 will change the tokenizer ID
    input_tokens_tt = [
        [
            ttnn.from_torch(
                input_tokens[:, i].unsqueeze(0), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            for device in devices
        ]
        for i in range(max_prompt_len)
    ]
    input_mask_tt = [
        [
            ttnn.from_torch(
                input_mask[:, i].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            for device in devices
        ]
        for i in range(max_prompt_len)
    ]
    return input_tokens_tt, max_prompt_len, input_mask_tt, input_tokens, input_mask_bool


@torch.no_grad()
def run_mixtral_demo(user_input, batch_size, devices, instruct_mode):
    assert batch_size == 32, "Batch size must be 32"

    dtype = ttnn.bfloat8_b

    embed_on_host = True  # Do embedding and argmax on host. TODO Seeing bad output when on device
    seqlen = 1  # Generating one token per user at a time

    logger.info(f"Reading inputs...")
    if len(user_input) == 1:
        input_prompts = user_input * 32  # Always process 32 users
    else:
        input_prompts = load_inputs(user_input, 32)

    # Load model args, weights, and tokenizer
    model_args = TtModelArgs(devices[0], instruct=instruct_mode)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    model_args.n_layers = 32  # Full model

    logger.info("Loading weights...")
    state_dict = torch.load(model_args.state_dict_path)
    # If not using the full model, remove the layers that are not used
    keys_dict = list(state_dict.keys())[:]
    remv = [f"layers.{i}" for i in range(model_args.n_layers, 32)]
    for k in keys_dict:
        if any([r in k for r in remv]):
            state_dict.pop(k)

    # Embedding on host
    if embed_on_host:
        embd = Emb()
        embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    logger.info("Loading weights finished!")

    # Preprocess initial prompt inputs
    input_tokens_tt, max_prompt_len, input_mask, input_tokens_pt, input_mask_pt = preprocess_inputs(
        input_prompts, tokenizer, model_args, dtype, instruct_mode, devices
    )

    # TODO should we just change the pad after initial pad of the inputs?
    if instruct_mode:
        tokenizer._model.pad_id = tokenizer._model.eos_id

    # Load TTNN mixtral model
    logger.info("Loading weights to device...")
    tt_model = TtTransformer(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    if not embed_on_host:
        tt_embds = [
            TtMixtralEmbedding(
                device=devices[i],
                args=model_args,
                weight_cache_path=model_args.weight_cache_path(dtype),
                state_dict=state_dict,
                dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
            )
            for i in range(len(devices))
        ]

    logger.info("Finished loading weights to device.")

    # Prepare the first token embedding for each user
    if embed_on_host:
        pt_decode_input = embd(input_tokens_pt[:, 0]).view(batch_size, seqlen, -1)
    else:  # Embedding on device
        # Each device does its own embedding
        decode_input_11BH = [tt_embds[i](input_tokens_tt[0][i]) for i in range(len(devices))]
        # Reshape and change row major to tile layout
        decode_input_11BH = [
            ttnn.reshape(decode_input_11BH[i], ttnn.Shape([1, 1, batch_size, model_args.dim]))
            for i in range(len(devices))
        ]
        decode_input_11BH = [ttnn.to_layout(decode_input_11BH[i], layout=ttnn.TILE_LAYOUT) for i in range(len(devices))]
        # decode_input_11BH = [ttnn.experimental.tensor.tilize(decode_input_11BH[i]) for i in range(len(devices))]
        # decode_input_11BH = [ttnn.experimental.tensor.tilize_with_val_padding(decode_input_11BH[i], ) for i in range(len(devices))]")

    # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
    rot_mats = prepare_rotation_mat_ttnn(
        model_args.head_dim,
        model_args.max_seq_len,
        tt_model.devices,
    )

    generation_start_pos = 0
    max_generated_tokens = 200  # TODO Increase to around 100 tokens

    cache_attention(devices, state_dict, model_args, rot_mats, generation_start_pos, max_generated_tokens, dtype)

    logger.info("Starting inference...")

    # Keep track of generated outputs to print out every iteration
    all_outputs = [[] for _ in range(batch_size)]

    # Keep running inference as long as there is a user in the batch still decoding or max tokens per user are decoded
    for iteration in range(max_generated_tokens):
        iteration_time_start = time()
        start_pos = generation_start_pos + iteration
        current_pos = start_pos % model_args.sliding_window

        if embed_on_host:
            decode_input_11BH = prepare_inputs_ttnn(
                pt_decode_input,
                model_args.dim,
                tt_model.devices,
            )

        # Run ttnn mixtral model
        tt_out_11BH = tt_model(decode_input_11BH, start_pos, current_pos, rot_mats)

        if embed_on_host:
            # Convert ttnn tensor to torch tensor
            tt_output_torch = ttnn.to_torch(tt_out_11BH[0]).squeeze(1).view(batch_size, seqlen, -1).detach().float()
            # tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
            # Argmax on host to get the new generated tokens
            tt_token_batch = sample(tt_output_torch, temperature=0, top_p=0.8)
            # Update the users that are still in prefill and the ones generating new tokens
            if iteration < max_prompt_len:
                tt_token_batch = torch.where(
                    input_mask_pt[:, iteration], input_tokens_pt[:, iteration], tt_token_batch[:, 0]
                ).unsqueeze(1)
            # Next PT input embedding
            pt_decode_input = embd(tt_token_batch).view(batch_size, seqlen, -1)
        else:  # Embedding/argmax on device
            for i in range(len(devices)):
                # TODO Update argmax to ttnn when OP becomes available
                tt_out_B11B = ttnn.experimental.tensor.argmax(tt_out_11BH[i], dim=-1)
                tt_out_1B = ttnn.reshape(tt_out_B11B[:1, :, :, :], ttnn.Shape([1, batch_size]))  # [1, 32] Bfloat16
                # Update the users that are still in prefill and the ones generating new tokens
                if iteration < max_prompt_len:
                    decode_input_1B = ttnn.where(input_mask[iteration][i], input_tokens_tt[iteration][i], tt_out_1B)
                else:
                    decode_input_1B = tt_out_1B

                # Next TT input embeddings
                decode_input_1BH = tt_embds[i](decode_input_1B)
                decode_input_11BH[i] = ttnn.reshape(decode_input_1BH, ttnn.Shape([1, 1, batch_size, model_args.dim]))
                decode_input_11BH[i] = ttnn.to_layout(decode_input_11BH[i], layout=ttnn.TILE_LAYOUT)

            # Convert ttnn tensor to torch tensor and print decoded output (from a single device)
            # tt_output_torch = ttnn.to_torch(decode_input_1B).transpose(0, 1)
            tt_token_batch = ttnn.to_torch(decode_input_1B).transpose(0, 1)

        # Get the generated tokens for each user for printing in the log
        for user in range(batch_size):
            user_tok = int(tt_token_batch[user].item())
            if user_tok != tokenizer.eos_id:  # Stop saving the ouput after hitting the EOS token
                all_outputs[user].append(user_tok)

        iteration_time = time() - iteration_time_start
        tokens_per_second_per_user = 1 / iteration_time
        # Print out generated outputs for each user at the end of every iteration
        if len(user_input) == 1:
            logger.info("[User 0] {}".format("".join(tokenizer.decode(all_outputs[0]))))
        else:
            for user in range(batch_size):
                logger.info("[User {}] {}".format(user, "".join(tokenizer.decode(all_outputs[user]))))

        logger.info(
            f"Iteration {iteration}: {1000*iteration_time:.2f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
        )


@pytest.mark.timeout(10000)
@pytest.mark.parametrize(
    "input_prompts, instruct_weights",
    [
        ("models/demos/t3000/mixtral8x7b/demo/input_data.json", False),
        ("models/demos/t3000/mixtral8x7b/demo/input_data_questions.json", True),
    ],
    ids=["general_weights", "instruct_weights"],
)
def test_mixtral8x7b_demo(all_devices, use_program_cache, input_prompts, instruct_weights):
    devices = get_devices_for_t3000(all_devices, 8)
    return run_mixtral_demo(user_input=input_prompts, batch_size=32, devices=devices, instruct_mode=instruct_weights)

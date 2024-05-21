# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.reference.llama.llama.model import precompute_freqs_cis, apply_rotary_emb
from models.experimental.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.experimental.llama2_70b.tt.llama_common import (
    get_llama_path,
)

from models.experimental.llama2_70b.tt.llama_common import precompute_freqs, freqs_to_rotation_matrix, gather_rotary_emb


def get_rotation_mat(dhead, end, start_pos, seqlen, batch):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


def get_rotation_mat_prefill(dhead, end, start_pos, seqlen, batch):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(batch, seqlen, dtype=torch.long) * torch.arange(start_pos, start_pos + seqlen).unsqueeze(
        0
    )
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


class TtLlamaRotary(torch.nn.Module):
    def __init__(
        self,
        device,
        hidden_size: int,
        n_heads,
        n_kv_heads,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads
        self.device = device

        tile_width = 32

        self.transformation_mat = torch2tt_tensor(get_rot_transformation_mat(dhead=tile_width), device)

    def apply_rotary(self, x, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K

        # x.shape = (1, 8, 128, 128)
        batch, n_heads, _, _ = x.shape
        rotary_output = tt_lib.tensor.rotary_embedding_llama(x, cos, sin, self.transformation_mat)

        return rotary_output

    # def chunk_to_tile(self, x):
    #     batch, n_heads, seq_len, head_dim = x.shape
    #     tile_width = 32
    #     x_chunked = torch.chunk(x, seq_len // tile_width, dim=2)
    #     x_chunked = [torch.chunk(x_row, head_dim // tile_width, dim=3) for x_row in x_chunked]
    #     return x_chunked

    # def apply_rotary(self, x, cos, sin):
    #     batch, n_heads, _, _ = x.shape

    #     # n_head = 8 for Q
    #     # n_head = 1 for K

    #     # x.shape = (1, 8, 128, 128)
    #     # cos = ttnn.repeat(cos, ttnn.Shape([batch, n_heads, 1, 1]))
    #     # sin = ttnn.repeat(sin, ttnn.Shape([batch, n_heads, 1, 1]))

    #     x = tt2torch_tensor(x)
    #     cos = tt2torch_tensor(cos)
    #     sin = tt2torch_tensor(sin)

    #     x_chunked = self.chunk_to_tile(x)
    #     cos_chunked = self.chunk_to_tile(cos)
    #     sin_chunked = self.chunk_to_tile(sin)

    #     x_transformed = torch.zeros(x.shape, dtype=torch.bfloat16)

    #     trans_mat = tt2torch_tensor(self.transformation_mat)

    #     x_cos = torch.zeros(x.shape, dtype=torch.bfloat16)
    #     x_sin = torch.zeros(x.shape, dtype=torch.bfloat16)
    #     rotary_output = torch.zeros(x.shape, dtype=torch.bfloat16)

    #     tile_size = 32

    #     for i in range(len(x_chunked)):
    #         for j in range(len(x_chunked[i])):
    #             row_window = slice(i * tile_size, (i + 1) * tile_size)
    #             col_window = slice(j * tile_size, (j + 1) * tile_size)

    #             x_transformed[:, :, row_window, col_window] = torch.matmul(x_chunked[i][j], trans_mat)
    #             x_cos[:, :, row_window, col_window] = torch.mul(cos_chunked[i][j], x_chunked[i][j])
    #             x_sin[:, :, row_window, col_window] = torch.mul(
    #                 sin_chunked[i][j], x_transformed[:, :, row_window, col_window]
    #             )
    #             rotary_output[:, :, row_window, col_window] = torch.add(
    #                 x_cos[:, :, row_window, col_window], x_sin[:, :, row_window, col_window]
    #             )

    #     rotary_output = torch2tt_tensor(rotary_output, self.device)
    #     return rotary_output

    def forward(self, xq, xk, cos, sin):
        xq = self.apply_rotary(xq, cos, sin)
        xk = self.apply_rotary(xk, cos, sin)
        return xq, xk


class PytorchLlamaRotaryModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.n_heads = hf_reference_model.params.n_heads
        self.n_kv_heads = hf_reference_model.params.n_kv_heads
        self.head_dim = hf_reference_model.params.dim // self.n_heads

    def forward(self, xq, xk, freqs_cis):
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        return xq, xk


def get_rot_transformation_mat(dhead):
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def compute_gather_cos_sin(dhead, end, position_ids):
    cos, sin = precompute_freqs(dhead, end)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def run_test_LlamaReshape(
    devices,
    batch,
    seq_len,
    pcc,
    model_config,
):
    # Prepare paths and devices
    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(devices, model_config, n_devices=1, emulated=True)
    device = devices[0]

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, max_seq_len=4096, max_batch_size=1, n_layers=1, skip_model_load=False
    ).model

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    inp = [
        (torch.rand(batch, 8, seq_len, head_dim) * 2) - 1,
        (torch.rand(batch, 1, seq_len, head_dim) * 2) - 1,
    ]
    freqs_cis = precompute_freqs_cis(
        # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
        # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        hidden_dim // n_heads,
        configuration.max_seq_len * 2,
    )  # torch.Size([8192, 64])

    start_pos = 0  # Must pick non-zero start pos to get non-zero freqs_cis
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
    # freqs_cis = freqs_cis.expand(batch, -1)  # torch.Size([32, 64])
    inp.append(freqs_cis)

    layer_num = 0
    base_url = "layers"
    # PyTorch Ground Truth output --------------------------------------------------------------------
    pytorch_model = PytorchLlamaRotaryModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_model(*inp)

    # TT hardware / Modified PyTorch execution -------------------------------------------------------------
    tt_model = TtLlamaRotary(
        device,
        hidden_dim,
        n_heads,
        n_kv_heads,
    )

    cos, sin = compute_gather_cos_sin(
        dhead=head_dim, end=configuration.max_seq_len * 2, position_ids=torch.arange(start_pos, start_pos + seq_len)
    )
    tt_inp = [inp[0], inp[1], cos, sin]
    tt_inp = [torch2tt_tensor(i, device) for i in tt_inp]

    tt_out = tt_model(*tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]

    # check outputs ----------------------------------------------------------------------
    does_pass = True
    for i in range(2):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], pcc)
        # Check each shape matches
        assert pytorch_out[i].shape == tt_out[i].shape
        logger.info(f"PCC value: {output_pcc}")
        does_pass = does_pass and out_pass

        mae = torch.mean(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"MAE: {mae}")

        max_incorrect = torch.max(torch.abs(pytorch_out[i] - tt_out[i]))
        logger.info(f"Max incorrect: {max_incorrect}")

        max_gt = torch.max(torch.abs(pytorch_out[i]))
        logger.info(f"Max ground truth: {max_gt}")

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    (
        (1, 128),
        (1, 2048),
    ),
    ids=(
        "prefill_128",
        "prefill_2k",
    ),
)
@pytest.mark.parametrize("pcc", (0.9997,))
def test_LlamaAttention_inference(
    batch,
    seq_len,
    pcc,
    all_devices,
):
    devices = all_devices
    model_config = get_model_config(num_devices=8, seq_len=seq_len)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")
    run_test_LlamaReshape(
        devices,
        batch,
        seq_len,
        pcc,
        model_config,
    )

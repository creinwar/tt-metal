# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor

import pytest


def sdpa(q, k, v, attn_mask):
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = scores / (q.size(-1) ** 0.5)
    scores = scores + attn_mask
    attn = torch.nn.functional.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def fa2(q, k, v, attn_mask):
    chunk_size = 256
    num_batches = q.size(0)
    num_heads = q.size(1)
    seq_len = q.size(2)
    num_chunks = seq_len // chunk_size

    scale = 1.0 / (q.size(-1) ** 0.5)

    out = torch.zeros_like(q)

    for b in range(num_batches):
        for h in range(num_heads):
            for q_c in range(num_chunks):
                # Fetch Q chunk
                q_chunk = q[b, h, q_c * chunk_size : (q_c + 1) * chunk_size]

                # Setup variables for the Q chunk
                cur_m, prev_m = torch.full((chunk_size, 1), -1000), torch.full((chunk_size, 1), -1000)
                cur_sum, prev_sum = torch.full((chunk_size, 1), 0), torch.full((chunk_size, 1), 0)
                chunk_output = torch.zeros_like(q_chunk)

                for k_c in range(num_chunks):
                    # MQA: only 1 kv head

                    # Fetch K, V, attn_mask chunks
                    k_chunk = k[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    v_chunk = v[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    attn_mask_chunk = attn_mask[
                        b, h, q_c * chunk_size : (q_c + 1) * chunk_size, k_c * chunk_size : (k_c + 1) * chunk_size
                    ]

                    scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
                    scores = scores * scale
                    scores = scores + attn_mask_chunk

                    # 9
                    rowmax = torch.max(scores, dim=-1, keepdim=True)[0]
                    cur_m = torch.maximum(prev_m, rowmax)
                    pij = torch.exp(scores - cur_m)
                    row_sum = torch.sum(pij, dim=-1, keepdim=True)

                    exp_max_diff = torch.exp(prev_m - cur_m)

                    cur_sum = exp_max_diff * prev_sum + row_sum

                    # 10
                    chunk_output = chunk_output * exp_max_diff + torch.matmul(pij, v_chunk)

                    prev_sum = cur_sum
                    prev_m = cur_m

                # 12
                chunk_output = chunk_output / cur_sum
                out[b, h, q_c * chunk_size : (q_c + 1) * chunk_size] = chunk_output

    return out


def fa2_cb(q, k, v, attn_mask):
    chunk_size = 256
    num_batches = q.size(0)
    num_heads = q.size(1)
    seq_len = q.size(2)
    num_chunks = seq_len // chunk_size

    scale = 1.0 / (q.size(-1) ** 0.5)

    out = torch.zeros_like(q)

    for b in range(num_batches):
        for h in range(num_heads):
            for q_c in range(num_chunks):
                # Fetch Q chunk
                cb_q_in = q[b, h, q_c * chunk_size : (q_c + 1) * chunk_size]

                # Setup variables for the Q chunk
                cb_cur_max, cb_prev_max = torch.full((chunk_size, 1), -1000), torch.full((chunk_size, 1), -1000)
                cb_cur_sum, cb_prev_sum = torch.full((chunk_size, 1), 0.0), torch.full((chunk_size, 1), 0.0)

                for k_c in range(num_chunks):
                    # MQA: only 1 kv head

                    # Fetch K, V, attn_mask chunks
                    cb_k_in = k[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    cb_v_in = v[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    cb_mask_in = attn_mask[
                        b, h, q_c * chunk_size : (q_c + 1) * chunk_size, k_c * chunk_size : (k_c + 1) * chunk_size
                    ]

                    cb_qk_im = torch.matmul(cb_q_in, cb_k_in.transpose(-1, -2))
                    cb_qk_im *= scale
                    cb_qk_im += cb_mask_in

                    # 9
                    cb_cur_max = torch.max(cb_qk_im, dim=-1, keepdim=True)[0]
                    cb_cur_max = torch.maximum(cb_prev_max, cb_cur_max)
                    cb_qk_im -= cb_cur_max  # eltwise sub bcast cols
                    cb_qk_im = torch.exp(cb_qk_im)
                    cb_cur_sum = torch.sum(cb_qk_im, dim=-1, keepdim=True)

                    cb_exp_max_diff = cb_prev_max - cb_cur_max
                    cb_exp_max_diff = torch.exp(cb_exp_max_diff)

                    cb_prev_sum *= cb_exp_max_diff
                    cb_cur_sum += cb_prev_sum

                    cb_out_im = torch.matmul(cb_qk_im, cb_v_in)

                    # 10
                    if k_c == 0:
                        cb_out_accumulate_im = cb_out_im
                    else:
                        cb_out_accumulate_im *= cb_exp_max_diff
                        cb_out_accumulate_im += cb_out_im

                    cb_prev_sum = cb_cur_sum
                    cb_prev_max = cb_cur_max

                # 12
                cb_cur_sum = 1.0 / cb_cur_sum
                cb_out_accumulate_im *= cb_cur_sum
                out[b, h, q_c * chunk_size : (q_c + 1) * chunk_size] = cb_out_accumulate_im

    return out


def fa2_fake(q, k, v, attn_mask):
    chunk_size = 256
    num_batches = q.size(0)
    num_heads = q.size(1)
    seq_len = q.size(2)
    num_chunks = seq_len // chunk_size

    scale = 1.0 / (q.size(-1) ** 0.5)

    out = torch.zeros_like(q)

    for b in range(num_batches):
        for h in range(num_heads):
            for q_c in range(num_chunks):
                # Fetch Q chunk
                cb_q_in = q[b, h, q_c * chunk_size : (q_c + 1) * chunk_size]

                # Setup variables for the Q chunk
                cb_cur_max, cb_prev_max = torch.full((chunk_size, 1), -1000), torch.full((chunk_size, 1), -1000)
                cb_cur_sum, cb_prev_sum = torch.full((chunk_size, 1), 0.0), torch.full((chunk_size, 1), 0.0)

                for k_c in range(num_chunks):
                    # MQA: only 1 kv head

                    # Fetch K, V, attn_mask chunks
                    cb_k_in = k[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    cb_v_in = v[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    cb_mask_in = attn_mask[
                        b, h, q_c * chunk_size : (q_c + 1) * chunk_size, k_c * chunk_size : (k_c + 1) * chunk_size
                    ]

                    cb_qk_im = torch.matmul(cb_q_in, cb_k_in.transpose(-1, -2))
                    cb_qk_im *= scale
                    cb_qk_im += cb_mask_in

                    # # 9
                    cb_cur_max = torch.max(cb_qk_im, dim=-1, keepdim=True)[0]
                    cb_cur_max = torch.maximum(cb_prev_max, cb_cur_max)
                    cb_qk_im -= cb_cur_max
                    cb_qk_im = torch.exp(cb_qk_im)
                    cb_cur_sum = torch.sum(cb_qk_im, dim=-1, keepdim=True)

                    cb_exp_max_diff = cb_prev_max - cb_cur_max
                    cb_exp_max_diff = torch.exp(cb_exp_max_diff)

                    cb_prev_sum *= cb_exp_max_diff
                    cb_cur_sum += cb_prev_sum

                    cb_out_im = torch.matmul(cb_qk_im, cb_v_in)

                    # 10
                    if k_c == 0:
                        cb_out_accumulate_im = cb_out_im
                    else:
                        cb_out_accumulate_im *= cb_exp_max_diff
                        cb_out_accumulate_im += cb_out_im

                    cb_prev_sum = cb_cur_sum
                    cb_prev_max = cb_cur_max

                # 12
                cb_cur_sum = 1.0 / cb_cur_sum
                cb_out_accumulate_im *= cb_cur_sum
                out[b, h, q_c * chunk_size : (q_c + 1) * chunk_size] = cb_out_accumulate_im

    return out


def tt_fa2(device, q, k, v, attn_mask, program_config, dtype):
    tt_q = torch2tt_tensor(q, device, tt_dtype=dtype)
    tt_k = torch2tt_tensor(k, device, tt_dtype=dtype)
    tt_v = torch2tt_tensor(v, device, tt_dtype=dtype)
    tt_attn_mask = torch2tt_tensor(attn_mask, device, tt_dtype=dtype)

    compute_kernel_config = tt_lib.tensor.WormholeComputeKernelConfig(
        math_fidelity=tt_lib.tensor.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_out = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        tt_attn_mask,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    return tt2torch_tensor(tt_out)


def tt_fa2_noconvert(device, q, k, v, attn_mask, program_config):
    tt_out = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
        q, k, v, attn_mask, is_causal=True, program_config=program_config
    )

    return tt2torch_tensor(tt_out)


def run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 8], q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size
    )

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    # attn_mask = torch.full((s, s), -10.0) #TODO: While exp is debugged
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)

    # Print shapes of all inputs along with input names
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print(f"attn_mask: {attn_mask.shape}")

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)
    mine = tt_fa2(device, Q, K, V, attn_mask, program_config, dtype)
    out_pass, out_pcc = comp_pcc(gt, mine, 0.99)
    print(f"python vs pytorch: {out_pcc}")

    assert out_pass


def run_test_sdpa_python():
    b = 1
    nh = 8
    nkv = 1
    s = 2048
    d = 128

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, nh, -1, -1)

    # Print shapes of all inputs along with input names
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print(f"attn_mask: {attn_mask.shape}")

    # is_causal must be false if we specify an attention_mask
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    mine = sdpa(Q, K, V, attn_mask)

    out_pass, out_pcc = comp_pcc(gt, mine, 0.99)
    print(f"python vs pytorch: {out_pcc}")
    assert out_pass

    fa = fa2(Q, K, V, attn_mask)
    out_pass, out_pcc = comp_pcc(gt, fa, 0.99)
    print(f"fa2 vs pytorch: {out_pcc}")
    assert out_pass


def run_test_sdpa_cb_python():
    b = 1
    nh = 8
    nkv = 1
    s = 2048
    d = 128

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, nh, -1, -1)

    # Print shapes of all inputs along with input names
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print(f"attn_mask: {attn_mask.shape}")

    # is_causal must be false if we specify an attention_mask
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    mine = fa2_cb(Q, K, V, attn_mask)

    out_pass, out_pcc = comp_pcc(gt, mine, 0.99)
    print(f"python vs pytorch: {out_pcc}")
    assert out_pass

    fa = fa2(Q, K, V, attn_mask)
    out_pass, out_pcc = comp_pcc(gt, fa, 0.99)
    print(f"fa2 vs pytorch: {out_pcc}")
    assert out_pass


def run_stress_sdpa_tt(device):
    b = 1
    nh = 8
    nkv = 1
    s = 2048
    d = 128

    device.enable_program_cache()

    program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 8], q_chunk_size=256, k_chunk_size=256
    )

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    # attn_mask = torch.full((s, s), -10.0) #TODO: While exp is being debugged
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, 1, -1, -1)
    # FOR DEBUG, don't use neginf

    # Print shapes of all inputs along with input names
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print(f"attn_mask: {attn_mask.shape}")

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask)

    tt_q = torch2tt_tensor(Q, device)
    tt_k = torch2tt_tensor(K, device)
    tt_v = torch2tt_tensor(V, device)
    tt_attn_mask = torch2tt_tensor(attn_mask, device)

    first_pcc = 0.0
    for i in range(1000):
        print(f"Iteration {i}")
        mine = tt_fa2_noconvert(device, tt_q, tt_k, tt_v, tt_attn_mask, program_config)
        out_pass, out_pcc = comp_pcc(gt, mine, 0.99)
        if i == 0:
            first_pcc = out_pcc
        else:
            # assert out_pcc == first_pcc, "ND PCC found"
            pass

        print(f"python vs pytorch: {out_pcc}")
        # assert out_pass


def test_sdpa_stress_tt(device):
    # device.enable_program_cache()
    run_stress_sdpa_tt(device)


def test_sdpa_cb_python():
    run_test_sdpa_cb_python()


def test_sdpa_python():
    run_test_sdpa_python()


@pytest.mark.parametrize(
    "dtype", [tt_lib.tensor.DataType.BFLOAT8_B, tt_lib.tensor.DataType.BFLOAT16], ids=["bfp8", "bf16"]
)
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    (
        [1, 8, 1, 2048, 128],  # Llama2-70B
        [1, 16, 1, 2048, 64],  # Falcon-40B
        [1, 71, 1, 2048, 64],  # Falcon-7B
        [32, 8, 1, 2048, 128],  # Llama2-70B large batch
        [1, 8, 1, 8192, 128],  # Llama2-70B large sequence
    ),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)

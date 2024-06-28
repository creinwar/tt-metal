# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from typing import Optional
from functools import partial

import tt_lib
import ttnn
from models.demos.metal_BERT_large_11.tt.mha import TtMultiHeadAttentionModel
from models.demos.metal_BERT_large_11.tt.ffn import TtFeedForwardModel
from models.demos.metal_BERT_large_11.tt import custom_matmuls
from tt_lib.utils import pad_weight


class TtBertEncoder:
    def __init__(self, config, encoder_idx, state_dict, device, model_config, tt_cache_path):
        self.device = device
        self.model_config = model_config

        # MHA sub-graph
        self.mha = TtMultiHeadAttentionModel(config, encoder_idx, state_dict, device, model_config, tt_cache_path)

        attn_layer_name = f"bert.encoder.layer.{encoder_idx}.attention.output"
        layer_name = f"bert.encoder.layer.{encoder_idx}.output"

        if tt_cache_path is not None:
            self.attention_output_weight = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.dense.weight_{self.model_config['OP7_SELFOUT_WEIGHTS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP7_SELFOUT_WEIGHTS_MEMCFG"])
            self.attention_output_bias = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.dense.bias_{self.model_config['OP7_SELFOUT_BIAS_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP7_SELFOUT_BIAS_MEMCFG"])
            self.mha_gamma = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.LayerNorm.weight_{self.model_config['OP8_LAYERNORM_GAMMA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP8_LAYERNORM_GAMMA_MEMCFG"])
            self.mha_beta = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{attn_layer_name}.LayerNorm.bias_{self.model_config['OP8_LAYERNORM_BETA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP8_LAYERNORM_BETA_MEMCFG"])
            self.ffn_gamma = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.LayerNorm.weight_{self.model_config['OP11_LAYERNORM_GAMMA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP8_LAYERNORM_GAMMA_MEMCFG"])
            self.ffn_beta = tt_lib.tensor.load_tensor(
                str(
                    tt_cache_path
                    / f"{layer_name}.LayerNorm.bias_{self.model_config['OP11_LAYERNORM_BETA_DTYPE'].name}.bin"
                )
            ).to(device, self.model_config["OP8_LAYERNORM_BETA_MEMCFG"])
        else:
            self.attention_output_weight = pad_weight(
                torch.transpose(
                    state_dict[f"{attn_layer_name}.dense.weight"],
                    -2,
                    -1,
                )
            )
            self.attention_output_weight = (
                tt_lib.tensor.Tensor(
                    self.attention_output_weight.reshape(-1).tolist(),
                    self.attention_output_weight.shape,
                    model_config["OP7_SELFOUT_WEIGHTS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP7_SELFOUT_WEIGHTS_MEMCFG"])
            )
            self.attention_output_bias = pad_weight(state_dict[f"{attn_layer_name}.dense.bias"])
            self.attention_output_bias = (
                tt_lib.tensor.Tensor(
                    self.attention_output_bias.reshape(-1).tolist(),
                    self.attention_output_bias.shape,
                    model_config["OP7_SELFOUT_BIAS_DTYPE"],
                    tt_lib.tensor.Layout.ROW_MAJOR,
                )
                .to(tt_lib.tensor.Layout.TILE)
                .to(device, model_config["OP7_SELFOUT_BIAS_MEMCFG"])
            )

            # Weights pre-transposed on host​. No on-the fly transpose of W.
            # self.attention_output_weight = tt_lib.tensor.transpose(
            #     self.attention_output_weight,
            #     -2, -1,
            # )

            # MHA layernorm
            gamma0 = state_dict[f"{attn_layer_name}.LayerNorm.weight"]
            beta0 = state_dict[f"{attn_layer_name}.LayerNorm.bias"]
            mha_gamma = gamma0.reshape(1, 1, -1, 32)
            self.mha_gamma = tt_lib.tensor.Tensor(
                mha_gamma.reshape(-1).tolist(),
                mha_gamma.shape,
                model_config["OP8_LAYERNORM_GAMMA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP8_LAYERNORM_GAMMA_MEMCFG"])
            mha_beta = beta0.reshape(1, 1, -1, 32)
            self.mha_beta = tt_lib.tensor.Tensor(
                mha_beta.reshape(-1).tolist(),
                mha_beta.shape,
                model_config["OP8_LAYERNORM_BETA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP8_LAYERNORM_BETA_MEMCFG"])

            # FFN layernorm
            gamma1 = state_dict[f"{layer_name}.LayerNorm.weight"]
            beta1 = state_dict[f"{layer_name}.LayerNorm.bias"]
            ffn_gamma = gamma1.reshape(1, 1, -1, 32)
            self.ffn_gamma = tt_lib.tensor.Tensor(
                ffn_gamma.reshape(-1).tolist(),
                ffn_gamma.shape,
                model_config["OP11_LAYERNORM_GAMMA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP11_LAYERNORM_GAMMA_MEMCFG"])
            ffn_beta = beta1.reshape(1, 1, -1, 32)
            self.ffn_beta = tt_lib.tensor.Tensor(
                ffn_beta.reshape(-1).tolist(),
                ffn_beta.shape,
                model_config["OP11_LAYERNORM_BETA_DTYPE"],
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(device, model_config["OP11_LAYERNORM_BETA_MEMCFG"])

        # FFN sub-graph
        self.ffn = TtFeedForwardModel(encoder_idx, state_dict, device, model_config, tt_cache_path)

        self.layer_norm_eps = config.layer_norm_eps

        if "OP7_SELFOUT_CONFIG" in model_config:

            def op7_mm_plus_bias(mha_res, attention_output_weight, attention_output_bias):
                mha_out = ttnn.linear(
                    mha_res,
                    attention_output_weight,
                    bias=attention_output_bias,
                    program_config=model_config["OP7_SELFOUT_CONFIG"],
                    memory_config=model_config["OP7_SELFOUT_OUTPUT_MEMCFG"],
                    dtype=model_config["OP7_SELFOUT_OUTPUT_DTYPE"],
                )
                return mha_out

        else:

            def op7_mm_plus_bias(mha_res, attention_output_weight, attention_output_bias):
                mha_out = custom_matmuls.bert_large_selfout_matmul(
                    mha_res,
                    attention_output_weight,
                    bias=attention_output_bias,
                    output_mem_config=model_config["OP7_SELFOUT_OUTPUT_MEMCFG"],
                    output_dtype=model_config["OP7_SELFOUT_OUTPUT_DTYPE"],
                )
                return mha_out

        self.op7_mm_plus_bias = op7_mm_plus_bias
        self.mha_ln_program_config = model_config.get(
            "OP8_LAYERNORM_CONFIG", tt_lib.operations.primary.LayerNormDefaultProgramConfig()
        )
        self.ffn_ln_program_config = model_config.get(
            "OP11_LAYERNORM_CONFIG", tt_lib.operations.primary.LayerNormDefaultProgramConfig()
        )

    def op8_add_layernorm(self, activation, mha_out):
        mha_out_add_and_norm = tt_lib.operations.primary.add_layernorm(
            activation,
            mha_out,
            self.layer_norm_eps,
            self.mha_gamma,
            self.mha_beta,
            program_config=self.mha_ln_program_config,
            output_mem_config=self.model_config["OP8_LAYERNORM_OUTPUT_MEMCFG"],
        )
        return mha_out_add_and_norm

    def op11_add_layernorm(self, mha_out_add_and_norm, ffn_out):
        ffn_out_add_and_norm = tt_lib.operations.primary.add_layernorm(
            mha_out_add_and_norm,
            ffn_out,
            self.layer_norm_eps,
            self.ffn_gamma,
            self.ffn_beta,
            program_config=self.ffn_ln_program_config,
            output_mem_config=self.model_config["OP11_LAYERNORM_OUTPUT_MEMCFG"],
        )
        return ffn_out_add_and_norm

    def __call__(
        self, activation: tt_lib.tensor.Tensor, attention_mask: Optional[tt_lib.tensor.Tensor] = None
    ) -> tt_lib.tensor.Tensor:
        # MHA - OP1 - OP6 ------------------------------->
        mha_res = self.mha(activation, attention_mask)
        # Don't deallocate activations here since it is used by more ops

        mha_out = self.op7_mm_plus_bias(mha_res, self.attention_output_weight, self.attention_output_bias)
        mha_res.deallocate()
        mha_out_add_and_norm = self.op8_add_layernorm(activation, mha_out)
        activation.deallocate()
        mha_out.deallocate()

        # FFN - OP9 - OP10 ----------------------------->
        ffn_out = self.ffn(mha_out_add_and_norm)

        ffn_out_add_and_norm = self.op11_add_layernorm(mha_out_add_and_norm, ffn_out)
        mha_out_add_and_norm.deallocate()
        ffn_out.deallocate()
        return ffn_out_add_and_norm

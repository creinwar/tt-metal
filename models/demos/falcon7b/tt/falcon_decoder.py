# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import ttnn
from models.demos.falcon7b.tt.falcon_attention import TtFalconAttentionDecode, TtFalconAttentionPrefill
from models.demos.falcon7b.tt.falcon_mlp import TtFalconMLPDecode, TtFalconMLPPrefill
from models.demos.falcon7b.tt.model_utils import get_weights_cached
from torch import nn
from models.utility_functions import tt2torch_tensor, torch2tt_tensor


class TtFalconDecoderLayer(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_dict = state_dict
        self.base_url = base_url
        self.devices = devices
        self.num_devices = len(devices)
        self.layer_num = layer_num
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config
        self.weights_dict = {}

        assert config.parallel_attn, "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn_prefill = TtFalconAttentionPrefill(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        self.self_attn_decode = TtFalconAttentionDecode(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        self.mlp_prefill = TtFalconMLPPrefill(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        self.mlp_decode = TtFalconMLPDecode(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            weights_dict=self.weights_dict,
        )

        layer_name = f"{base_url}.{layer_num}"

        layernorm_weights_str = f"{layer_name}.input_layernorm.weight"
        layernorm_bias_str = f"{layer_name}.input_layernorm.bias"

        self.lnweight = self.state_dict[layernorm_weights_str]
        self.lnbias = self.state_dict[layernorm_bias_str]

        self.layernorm_gamma = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            layernorm_weights_str,
            weight_config_str="INPUT_LAYERNORM_WEIGHTS",
            weights_to_cache=(self.state_dict[layernorm_weights_str] if self.state_dict else None),
            padzero=True,
            # padzero=True,
        )
        self.layernorm_beta = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            layernorm_bias_str,
            weight_config_str="INPUT_LAYERNORM_BIAS",
            weights_to_cache=(self.state_dict[layernorm_bias_str] if self.state_dict else None),
            padzero=True,
            # padzero=True,
        )

        # for i in range(self.num_devices):
        #     self.layernorm_gamma[i] = ttnn.experimental.tensor.repeat(
        #         self.layernorm_gamma[i],
        #         [1, 1, 32, 1],
        #         output_mem_config=self.model_config["INPUT_LAYERNORM_WEIGHTS_MEMCFG"],
        #     )
        # for i in range(self.num_devices):
        #     self.layernorm_beta[i] = ttnn.experimental.tensor.repeat(
        #         self.layernorm_beta[i],
        #         [1, 1, 32, 1],
        #         output_mem_config=self.model_config["INPUT_LAYERNORM_BIAS_MEMCFG"],
        #     )
        # for i in range(self.num_devices):
        #     self.layernorm_gamma[i] = ttnn.experimental.tensor.tilize(
        #         self.layernorm_gamma[i],
        #         output_mem_config=self.model_config["INPUT_LAYERNORM_WEIGHTS_MEMCFG"],
        #         output_dtype=self.model_config["INPUT_LAYERNORM_WEIGHTS_DTYPE"],
        #     )
        # for i in range(self.num_devices):
        #     self.layernorm_beta[i] = ttnn.experimental.tensor.tilize(
        #         self.layernorm_beta[i],
        #         output_mem_config=self.model_config["INPUT_LAYERNORM_BIAS_MEMCFG"],
        #         output_dtype=self.model_config["INPUT_LAYERNORM_BIAS_DTYPE"],
        #     )

        self.layernorm_eps = config.layer_norm_epsilon

    def forward(
        self,
        hidden_states: ttnn.experimental.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.experimental.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[ttnn.experimental.tensor.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        layernorm_output = []
        # layernorm_output2 = []
        # for i in range(self.num_devices):
        #     layernorm_output.append(tt2torch_tensor(hidden_states[i]))
        #     layernorm_output[i] = torch.nn.functional.layer_norm(layernorm_output[i], (4544,), weight=self.lnweight, bias=self.lnbias, eps=self.layernorm_eps)
        #     layernorm_output[i] = torch2tt_tensor(layernorm_output[i], self.devices[i])

        # ln_compute_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        #     math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
        #     math_approx_mode=True,
        #     fp32_dest_acc_en=True,
        #     packer_l1_acc=False,
        # )

        sharded_mem_cfg = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            ttnn.experimental.tensor.ShardSpec(
                ttnn.experimental.tensor.CoreRangeSet(
                    {
                        ttnn.experimental.tensor.CoreRange(
                            ttnn.experimental.tensor.CoreCoord(0, 0),
                            ttnn.experimental.tensor.CoreCoord(0, 1),
                        ),
                    }
                ),
                [
                    32,
                    4544 // 2,
                ],
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        # prog_cfg = ttnn.experimental.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        #     compute_with_storage_grid_size=[8,8],
        #     subblock_w=8,
        #     block_h=32 // 32 // 1,
        #     block_w=4544 // 32 // 2,
        #     inplace=True,
        # )

        ln_in = []

        for i in range(self.num_devices):
            ln_in.append(
                ttnn.experimental.tensor.interleaved_to_sharded(hidden_states[i], sharded_mem_config=sharded_mem_cfg)
            )

        breakpoint()

        for i in range(self.num_devices):
            layernorm_output.append(
                ttnn.experimental.tensor.layernorm(
                    ln_in[i],
                    self.layernorm_eps,
                    output_mem_config=sharded_mem_cfg,
                    # program_config=prog_cfg
                )
            )

        breakpoint()

        for i in range(self.num_devices):
            layernorm_output[i] = ttnn.experimental.tensor.sharded_to_interleaved(
                layernorm_output[i], output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"]
            )

        # for i in range(self.num_devices):
        #     layernorm_output.append(
        #         ttnn.experimental.tensor.layernorm(
        #             hidden_states[i],
        #             self.layernorm_eps,
        #             output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
        #             #compute_kernel_config = ln_compute_config
        #         )
        #     )

        self.out_ln_eps = [ttnn.experimental.tensor.clone(layernorm_output[i]) for i in range(self.num_devices)]

        # for i in range(self.num_devices):
        #     layernorm_output[i] = ttnn.experimental.tensor.mul(layernorm_output[i], self.layernorm_gamma[i], output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"])
        # for i in range(self.num_devices):
        #     layernorm_output[i] = ttnn.experimental.tensor.add(layernorm_output[i], self.layernorm_beta[i], output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"])
        for i in range(self.num_devices):
            layernorm_output[i] = ttnn.experimental.tensor.bcast(
                layernorm_output[i],
                self.layernorm_gamma[i],
                ttnn.experimental.tensor.BcastOpMath.MUL,
                ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
            )

        self.out_ln_g = [ttnn.experimental.tensor.clone(layernorm_output[i]) for i in range(self.num_devices)]

        for i in range(self.num_devices):
            layernorm_output[i] = ttnn.experimental.tensor.bcast(
                layernorm_output[i],
                self.layernorm_beta[i],
                ttnn.experimental.tensor.BcastOpMath.ADD,
                ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
            )

        residual = hidden_states

        # Attention and MLP execution
        # mlp will deallocate layernorm_output
        if llm_mode == "prefill":
            attn_outputs = self.self_attn_prefill(
                hidden_states=layernorm_output,
                alibi=alibi,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attention_output, layer_present = attn_outputs[0], attn_outputs[1]
            mlp_output = self.mlp_prefill(layernorm_output)

        elif llm_mode == "decode":
            attn_outputs = self.self_attn_decode(
                hidden_states=layernorm_output,
                alibi=alibi,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attention_output, layer_present = attn_outputs[0], attn_outputs[1]
            self.out_attn = [ttnn.experimental.tensor.clone(attention_output[i]) for i in range(self.num_devices)]
            mlp_output = self.mlp_decode(layernorm_output)
            self.out_mlp = [ttnn.experimental.tensor.clone(mlp_output[i]) for i in range(self.num_devices)]

        else:
            raise ValueError(f"Unknown llm_mode: {llm_mode}")

        output = []
        for i in range(self.num_devices):
            output.append(
                ttnn.add(
                    mlp_output[i],
                    attention_output[i],
                    memory_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
                )
            )
            mlp_output[i].deallocate()
            attention_output[i].deallocate()

        self.out_attnadd = [ttnn.experimental.tensor.clone(output[i]) for i in range(self.num_devices)]

        # dropout_add
        # For inference, this is just add
        for i in range(self.num_devices):
            output[i] = ttnn.add(
                output[i],
                residual[i],
                memory_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
            )
            residual[i].deallocate()

        self.out_resadd = [ttnn.experimental.tensor.clone(output[i]) for i in range(self.num_devices)]

        if use_cache:
            outputs = (output, layer_present)
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs

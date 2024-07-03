# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import nearest_32
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import LightweightModule


class TtMixtralAttention(LightweightModule):
    def __init__(self, device_mesh, state_dict, args, layer_num, dtype):
        super().__init__()
        self.num_devices = 8
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.model_args = args

        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.model_config = self.model_args.get_model_config()

        layer_name = f"layers.{layer_num}.attention"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: self.model_args.weight_cache_path(dtype) / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        self.wqkv = ttnn.as_tensor(
            torch.concat(
                [
                    torch.concat(
                        [
                            torch.transpose(
                                torch.chunk(self.state_dict[wq_str], self.num_devices)[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                torch.chunk(self.state_dict[wk_str], self.num_devices)[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                torch.chunk(self.state_dict[wv_str], self.num_devices)[i],
                                -2,
                                -1,
                            ),
                        ],
                        dim=-1,
                    )
                    for i in range(self.num_devices)
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=-1),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wqkv_multidevice_4d"),
        )

        self.wo = ttnn.as_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=-2),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wo_multidevice4d"),
        )

        cache_k = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.max_seq_len,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.max_seq_len,
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.as_tensor(
                lp,
                device=self.device_mesh,
                mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
                dtype=ttnn.bfloat8_b,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                memory_config=self.model_config["ATTN_CACHE_WEIGHTS_MEMCFG"],
                cache_file_name=cache_name(f"empty_attn_cache_{cache_k.shape}"),
            )
            for lp in layer_past
        ]

        self.scale = self.head_dim**-0.5

        reduce_mask_torch = torch.zeros(1, 1, self.max_batch_size, self.max_batch_size * 8)
        for i in range(self.max_batch_size):
            reduce_mask_torch[:, :, i, range(i, self.max_batch_size * 8, self.max_batch_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )

        self.compute_kernel = self.model_args.get_compute_kernel_config()
        self.compute_kernel_attn = self.model_args.get_compute_kernel_attn_config()

        self.core_grid = self.model_args.max_grid_size
        self.core_grid_attention = self.model_args.core_grid_attention

        # Will be filled during the initial warmup run
        self.q_mem_config = None
        self.k_mem_config = None

    def forward_decode(
        self,
        xs,
        start_pos,
        current_pos,
        attn_masks,
        rot_mat,
    ):
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        current_pos: start_pos
        attn_masks: (seq_len, batch, n_heads, cache_len+seq_len)
        rot_mats: list of rotation matrices for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        D : head_dim (128)
        P : padded_layer_past_len
        """
        padded_layer_past_len = nearest_32(start_pos + 1)

        x_11BH = xs
        wo = self.wo
        layer_past = self.layer_past
        rot_mat = rot_mat
        attn_mask_1B4P = attn_masks
        ###
        # QKV matmuls
        ###

        xqkv_fused = ttnn.matmul(
            x_11BH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            program_config=self.model_config["QKV_MM_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
        )

        # split qkv into heads
        (
            q_heads_1B4D,
            k_heads_1B1D,
            v_heads_1B1D,
        ) = ttnn.experimental.tensor.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###
        if self.q_mem_config is None:
            self.q_mem_config = q_heads_1B4D.memory_config()
        if self.k_mem_config is None:
            self.k_mem_config = k_heads_1B1D.memory_config()

        q_heads_1B4D = ttnn.matmul(
            q_heads_1B4D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=self.q_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )
        k_heads_1B1D = ttnn.matmul(
            k_heads_1B1D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=self.k_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        ###
        # KV update
        ###
        keys_1BPD = layer_past[0]
        values_1BPD = layer_past[1]
        ttnn.kv_cache.update_cache_for_token_(keys_1BPD, k_heads_1B1D, current_pos)
        ttnn.kv_cache.update_cache_for_token_(values_1BPD, v_heads_1B1D, current_pos)
        self.layer_past = [keys_1BPD, values_1BPD]
        k_heads_1B1D.deallocate(True)
        v_heads_1B1D.deallocate(True)

        keys_1BPD = ttnn.experimental.tensor.nlp_kv_cache_load_slice(
            keys_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        ###
        # Attention
        ###
        # transpose keys
        keys_1BDP = ttnn.experimental.tensor.transpose(
            keys_1BPD,
            -2,
            -1,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        keys_1BPD.deallocate(True)

        # scores matmul

        attn_1B4P = ttnn.matmul(
            q_heads_1B4D,
            keys_1BDP,
            dtype=ttnn.bfloat16,
            program_config=self.model_config["SCORES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            memory_config=self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"](padded_layer_past_len),
            compute_kernel_config=self.compute_kernel_attn,
        )
        q_heads_1B4D.deallocate(True)
        keys_1BDP.deallocate(True)

        # Softmax and scaling

        attn_1B4P = ttnn.experimental.operations.primary.transformers.scale_mask_softmax_in_place(
            attn_1B4P,
            self.scale,
            attn_mask_1B4P,
            program_config=self.model_config["ATTN_BATCHED_SOFTMAX_PROGCFG"](padded_layer_past_len),
            is_causal_mask=True,
        )

        # values matmul
        values_1BPD = ttnn.experimental.tensor.nlp_kv_cache_load_slice(
            values_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        attn_output_1B4D = ttnn.matmul(
            attn_1B4P,
            values_1BPD,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            program_config=self.model_config["VALUES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            compute_kernel_config=self.compute_kernel_attn,
        )
        attn_1B4P.deallocate(True)
        values_1BPD.deallocate(True)

        attn_output_11BH = ttnn.experimental.tensor.nlp_concat_heads_decode(
            attn_output_1B4D,
            num_heads=4,
        )
        attn_output_1B4D.deallocate(True)

        # attn_output_11BH = ttnn.experimental.tensor.sharded_to_interleaved(
        #     attn_output_11BH, output_mem_config=ttnn.L1_MEMORY_CONFIG
        # )

        ###
        # Output matmul
        ###

        dense_out_11BH = ttnn.matmul(
            attn_output_11BH,
            wo,
            memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
            # compute_with_storage_grid_size=(8, 8),
            program_config=self.model_config["LM_HEAD_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
        )
        attn_output_11BH.deallocate(True)
        # All gather
        dense_outputs_11BH = ttnn.all_gather(dense_out_11BH, dim=2, num_links=1)

        # return the sum of the outputs
        dense_outputs_11BH = ttnn.matmul(self.reduce_mask, dense_outputs_11BH)
        return dense_outputs_11BH

    def forward_prefill(self, xs_11SH, attn_masks, rot_mats, transformation_mats, user_id: int = 0):
        assert xs_11SH.shape[2] % 128 == 0 and xs_11SH.shape[2] > 0, "Seqlen must be divisible by 128"
        seq_len = xs_11SH.shape[-2]
        ###
        # QKV matmuls
        ###
        xqkv_program_config = None
        xqkv_mem_config = ttnn.L1_MEMORY_CONFIG
        if seq_len > 8192:
            xs_11SH = ttnn.reshape(xs_11SH, (1, seq_len // 2048, 2048, -1))
            xqkv_program_config = self.model_config["WQKV_PREFILL_PROGCFG"]
            xqkv_mem_config = ttnn.DRAM_MEMORY_CONFIG
        xqkv_fused = ttnn.linear(
            xs_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=xqkv_mem_config,  # self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not xqkv_program_config else None,
            compute_kernel_config=self.compute_kernel,
            program_config=xqkv_program_config,
        )

        xs_11SH.deallocate(True)

        if seq_len > 8192:
            xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, seq_len, -1))
        # split qkv into heads
        (
            q_heads_14SD_pre_rot,
            k_heads_11SD_pre_rot,
            v_heads_11SD,
        ) = ttnn.experimental.tensor.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            output_mem_config=ttnn.DRAM_MEMORY_CONFIG,  # self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )

        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###

        q_heads_14SD = ttnn.experimental.tensor.rotary_embedding_llama(
            q_heads_14SD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        q_heads_14SD_pre_rot.deallocate(True)

        k_heads_11SD = ttnn.experimental.tensor.rotary_embedding_llama(
            k_heads_11SD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        k_heads_11SD_pre_rot.deallocate(True)

        # FILL KV CACHE
        keys_11SD = self.layer_past[0]
        values_11SD = self.layer_past[1]
        keys_reshaped = ttnn.reshape(keys_11SD, [self.max_batch_size, self.n_local_kv_heads, -1, self.head_dim])
        values_reshaped = ttnn.reshape(values_11SD, [self.max_batch_size, self.n_local_kv_heads, -1, self.head_dim])
        ttnn.kv_cache.fill_cache_for_user_(
            keys_reshaped, ttnn.experimental.tensor.typecast(k_heads_11SD, dtype=ttnn.bfloat8_b), user_id
        )
        ttnn.kv_cache.fill_cache_for_user_(
            values_reshaped, ttnn.experimental.tensor.typecast(v_heads_11SD, dtype=ttnn.bfloat8_b), user_id
        )
        keys_11SD = ttnn.reshape(keys_reshaped, [self.n_local_kv_heads, self.max_batch_size, -1, self.head_dim])
        values_11SD = ttnn.reshape(values_reshaped, [self.n_local_kv_heads, self.max_batch_size, -1, self.head_dim])
        self.layer_past = [keys_11SD, values_11SD]

        # SDPA

        attn_output_14SD = ttnn.experimental.operations.primary.transformers.scaled_dot_product_attention(
            q_heads_14SD,
            k_heads_11SD,
            v_heads_11SD,
            attn_masks,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )

        # deallocate keys and values
        q_heads_14SD.deallocate(True)
        k_heads_11SD.deallocate(True)
        v_heads_11SD.deallocate(True)

        ###
        # Output matmul
        ###

        attn_output_11SH = ttnn.experimental.tensor.nlp_concat_heads(
            attn_output_14SD,
            output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output_14SD.deallocate(True)

        wo_program_config = None
        if seq_len > 2048:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, (1, seq_len // 2048, 2048, -1))
            wo_program_config = self.model_config["WO_PREFILL_PROGCFG"]

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not wo_program_config else None,
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=wo_program_config,
        )
        attn_output_11SH.deallocate(True)

        if seq_len > 2048:
            output_11SH = ttnn.reshape(output_11SH, (1, 1, seq_len, -1))
        output_11BH_gathered = ttnn.all_gather(output_11SH, dim=1, num_links=1)
        output_11SH.deallocate(True)
        output_11BH_reduced = ttnn.experimental.tensor.fast_reduce_nc(
            output_11BH_gathered, dims=[1], output=None, compute_kernel_config=None
        )
        output_11BH_gathered.deallocate(True)
        return output_11BH_reduced

    def forward(
        self, xs, start_pos, current_pos, attn_masks, rot_mats, transformation_mats=None, user_id=0, mode="decode"
    ):
        if mode == "prefill":
            return self.forward_prefill(xs, attn_masks, rot_mats, transformation_mats, user_id)
        else:
            return self.forward_decode(xs, start_pos, current_pos, attn_masks, rot_mats)

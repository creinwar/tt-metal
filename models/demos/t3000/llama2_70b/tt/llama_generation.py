# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ConcatMeshToTensor

from loguru import logger

import copy
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from models.demos.t3000.llama2_70b.tt.llama_common import BASE_URL
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, device_mesh, n_devices, n_layers, cache_path=None):
        ## Get state dict
        configuration = copy.deepcopy(configuration)

        # Cache Weights setup
        if n_layers == None:
            n_layers = 80

        model_config = get_model_config()

        # TT model -------------------------------------------------------------
        self.tt_model = TtLlamaModel(
            device_mesh,
            state_dict,
            BASE_URL,
            n_layers,
            model_config,
            configuration,
            cache_path=cache_path,
            read_cache=False,
        )
        self.params = configuration
        self.device_mesh = device_mesh
        self.n_devices = device_mesh.get_num_devices()

        # for device in devices:
        #     ttl.device.Synchronize(device)

        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        # First, determine whether this is decode or prefill based on shape of the input
        assert len(tokens.shape) == 2
        batch, seq_len = tokens.shape
        seq_len = 1 if kwargs.get("decode_only", False) else seq_len
        if seq_len == 1:
            # Decode
            # if current model config is not for decode, change it to decode
            if self.tt_model.model_config["LLM_MODE"] != "decode":
                logger.info("Changing mode to decode")
                model_config = get_model_config(batch=batch, seq_len=seq_len)
                self.tt_model.set_model_config(model_config)
            return self.decode_forward(tokens, start_pos, *args, **kwargs)
        else:
            # Prefill
            # if current model config is not for prefill, change it to prefill
            if self.tt_model.model_config["LLM_MODE"] != "prefill":
                logger.info("Changing mode to prefill")
                assert seq_len <= 2048, f"Only prefill up to 2048 tokens is supported, got {seq_len}"
                prefill_seq_len = 128 if seq_len <= 128 else 2048
                model_config = get_model_config(batch=batch, seq_len=prefill_seq_len)
                self.tt_model.set_model_config(model_config)
            return self.prefill_forward(tokens, start_pos, *args, **kwargs)

    def decode_forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(tokens, start_pos)

        ########### TRACE ###########
        import time

        logger.info(f"Trace with input shape: {tt_inp_emb.shape=}")

        logger.info("Compiling Model")
        c1 = time.time()
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
        )
        logits = ttnn.to_torch(
            tt_logits, device=self.device_mesh, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=3)
        )
        c2 = time.time()
        logger.info(f"Compiling Model took: {c2 - c1} seconds.")

        logger.info("Capturing Trace")
        t1 = time.time()
        trace_id = ttnn.begin_trace_capture(self.device_mesh, cq_id=0)
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
        )
        ttnn.end_trace_capture(self.device_mesh, trace_id, cq_id=0)
        t2 = time.time()
        logger.info(f"Capturing Trace took: {t2 - t1} seconds.")

        logger.info("Starting Trace perf test...")
        num_iters = 100

        times = []
        for i in range(num_iters):
            x1 = time.time()
            ttnn.execute_trace(self.device_mesh, trace_id, blocking=False)
            logits = ttnn.to_torch(
                tt_logits, device=self.device_mesh, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=3)
            )

            x2 = time.time()

            times.append(x2 - x1)
        logger.info(
            f"Ran Trace for {num_iters} iterations. Avg Trace execution time: {sum(times[1:]) / len(times[1:])} seconds."
        )
        print(times)
        ttnn.release_trace(self.device_mesh, trace_id)
        breakpoint()

        ########### TRACE ###########

        del tt_inp_emb
        del rot_mat
        del attn_mask

        # for device in self.devices:
        #     ttl.device.Synchronize(device)
        # logits = ttnn.from_device(tt_logits)
        logits = ttnn.to_torch(
            tt_logits, device=self.device_mesh, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=3)
        )

        # logits = torch.cat([tt2torch_tensor(tt_o) for tt_o in tt_logits], -1)
        logits = logits[..., : self.params.vocab_size].float()
        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        del tt_logits

        return logits

    def prefill_forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        batch, seq_len = tokens.shape
        output_logits = torch.zeros(batch, seq_len, self.params.vocab_size)
        padded_seq_len = 128 if seq_len <= 128 else 2048
        # pad tokens to 128 or 2048
        prefill_ids = torch.cat([tokens, torch.zeros(batch, padded_seq_len - seq_len).long()], dim=-1)

        for user_id in range(batch):
            logger.info(f"Filling kv cache for user {user_id + 1}")

            tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(
                prefill_ids[user_id : user_id + 1], start_pos=0, valid_seq_len=seq_len
            )

            tt_logits = self.tt_model(
                tt_inp_emb,
                rot_mat,
                start_pos,
                attn_mask,
                user_id=user_id,
            )

            del tt_inp_emb
            del rot_mat
            del attn_mask

            # for device in self.devices:
            #     ttl.device.Synchronize(device)

            logits = ttnn.to_torch(
                tt_logits, device=self.device_mesh, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=3)
            )
            logits = logits.squeeze(1)
            logits = logits[..., : self.params.vocab_size].float()  # [batch, seq_len, vocab_size]
            del tt_logits

            output_logits[user_id] = logits[:, :seq_len, :]

        logger.info(f"Finished prefill for all users up to {seq_len} tokens, Starting decode...")

        return output_logits

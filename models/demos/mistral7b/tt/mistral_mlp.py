# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMistralMLP(torch.nn.Module):
    def __init__(
        self,
        device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args
        self.model_config = model_config
        self.dtype = dtype

        base_name = f"layers.{layer_num}.feed_forward"
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{base_name}.{name}.weight"], -2, -1)
        cache_name = lambda name: weight_cache_path / (base_name + f".{name}")
        as_tensor = lambda name, type: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            device=self.device,
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1", ttnn.bfloat4_b)
        self.w2 = as_tensor("w2", ttnn.bfloat8_b)
        self.w3 = as_tensor("w3", ttnn.bfloat4_b)

        x_shape = ttnn.Shape([1, 1, args.max_batch_size, args.dim])
        h_shape = ttnn.Shape([1, 1, args.max_batch_size, args.hidden_dim])
        # self.w1_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
        #     input_shape_a=x_shape,
        #     input_shape_b=self.w1.shape,
        #     core_grid=self.args.max_grid_size,
        #     activation="silu",
        #     fp32_dst=self.args.get_compute_kernel_config().fp32_dest_acc_en,
        # )
        # self.w2_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
        #     input_shape_a=h_shape,
        #     input_shape_b=self.w2.shape,
        #     core_grid=self.args.max_grid_size,
        #     fp32_dst=self.args.get_compute_kernel_config().fp32_dest_acc_en,
        # )
        # self.w3_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
        #     input_shape_a=x_shape,
        #     input_shape_b=self.w3.shape,
        #     core_grid=self.args.max_grid_size,
        #     fp32_dst=self.args.get_compute_kernel_config().fp32_dest_acc_en,
        # )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        w1_out = ttnn.linear(
            x,
            self.w1,
            activation="silu",
            memory_config=self.model_config["FF1_OUTPUT_MEMCFG"],
            use_1d_systolic_array=True,
            # program_config=self.w1_program_config,
            compute_kernel_config=self.args.get_compute_kernel_config(),
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            memory_config=self.model_config["FF3_OUTPUT_MEMCFG"],
            use_1d_systolic_array=True,
            # program_config=self.w3_program_config,
            compute_kernel_config=self.args.get_compute_kernel_config(),
        )
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            memory_config=self.model_config["FF3_OUTPUT_MEMCFG"],
        )
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            memory_config=self.model_config["FF2_OUTPUT_MEMCFG"],
            use_1d_systolic_array=True,
            # program_config=self.w2_program_config,
            compute_kernel_config=self.args.get_compute_kernel_config(),
        )

        return w2_out

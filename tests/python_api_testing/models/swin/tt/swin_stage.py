from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import pickle

from python_api_testing.models.utility_functions_new import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from python_api_testing.models.swin.tt.swin_layer import TtSwinLayer
import tt_lib
from tt_lib.fallback_ops import fallback_ops


class TtSwinStage(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        downsample,
        state_dict,
        base_address,
        device,
        host,
        layer_index,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.device = device
        self.host = host
        self.layer_index = layer_index
        self.blocks = nn.ModuleList(
            [
                TtSwinLayer(
                    config=self.config,
                    dim=self.dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    state_dict=state_dict,
                    base_address=f"{base_address}.blocks.{i}",
                    device=self.device,
                    host=self.host,
                    shift_size=0 if i % 2 == 0 else self.config.window_size // 2,
                    layer_index=self.layer_index,
                    index=i,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                self.config,
                input_resolution,
                self.dim,
                state_dict,
                base_address,
                self.device,
                self.host,
            )
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[tt_lib.tensor.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            pt_swin_layer_input = tt_to_torch_tensor(hidden_states, self.host).squeeze(
                0
            )
            name = (
                "layer_"
                + str(self.layer_index)
                + "_tt_swin_layer_input_"
                + str(i)
                + ".pkl"
            )

            with open(name, "wb") as file:
                pickle.dump(pt_swin_layer_input, file)

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]
            pt_swin_layer_output = tt_to_torch_tensor(hidden_states, self.host).squeeze(
                0
            )
            name = (
                "layer_"
                + str(self.layer_index)
                + "_tt_swin_layer_output_"
                + str(i)
                + ".pkl"
            )

            with open(name, "wb") as file:
                pickle.dump(pt_swin_layer_output, file)

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(
                hidden_states_before_downsampling, input_dimensions
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs

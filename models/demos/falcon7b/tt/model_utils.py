# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.utility_functions import pad_by_zero, torch2tt_tensor


def get_weights_cached(
    devices,
    model_config,
    tt_cache_path,
    weight_cache_str,
    weight_config_str,
    weights_to_cache,
    overwrite=False,
    padzero=False,
    tt_layout=tt_lib.tensor.Layout.TILE,
    weights_dict=None,
    custom_output_shape=None,
):
    if padzero:
        assert tt_layout == tt_lib.tensor.Layout.TILE, "padding by zero currently only uses TILE layout"

    """Load weights from weights_dict or cache and duplicate per device. Store if not cached."""
    custom_output_shape_str = ""
    if custom_output_shape is not None:
        custom_output_shape_str = f"_{custom_output_shape[-2]}_{custom_output_shape[-1]}"
    path = (
        tt_cache_path
        / f"{weight_cache_str}_{model_config[f'{weight_config_str}_DTYPE'].name}{custom_output_shape_str}.bin"
    )

    if weights_dict and str(path) in weights_dict.keys():
        weights = weights_dict[str(path)]
    elif not overwrite and path.exists():
        # Load cached weights
        weights_host = tt_lib.tensor.load_tensor(str(path))
        # Duplicate weights on all devices
        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
        # Add to weights_dict
        if weights_dict is not None:
            weights_dict[str(path)] = weights
    else:
        if weights_to_cache is None:
            raise ValueError(f"weights_to_cache is None for {weight_cache_str}")
        
        if padzero:
            weights_host = pad_by_zero(
                weights_to_cache,
                device=None,
                tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
            )[0]
        else:
            if custom_output_shape is not None:
                padding = (
                    0,
                    custom_output_shape[-1] - weights_to_cache.shape[-1],
                    0,
                    custom_output_shape[-2] - weights_to_cache.shape[-2],
                )
                weights_to_cache = torch.nn.functional.pad(weights_to_cache, padding, "constant", 0.0)
            
            weights_host = torch2tt_tensor(
                weights_to_cache,
                tt_device=None,
                tt_layout=tt_layout,
                tt_memory_config=model_config[f"{weight_config_str}_MEMCFG"],
                tt_dtype=model_config[f"{weight_config_str}_DTYPE"],
            )
            
            # Store weights
            tt_lib.tensor.dump_tensor(str(path), weights_host)

        weights = [weights_host.to(device, model_config[f"{weight_config_str}_MEMCFG"]) for device in devices]
        # Save weights for reuse between prefill/decode
        if weights_dict is not None:
            weights_dict[str(path)] = weights[0]
            
    return weights

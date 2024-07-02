# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_op import TTPyOp
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_untilize_with_halo import TTPyUntilizeWithHalo
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    determine_parallel_config,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_utils import (
    SlidingWindowOpParamsWithParallelConfig,
    SlidingWindowOpParams,
    get_hash_from_sliding_window_op_params,
    calculate_shard_grid,
    calculate_memory_config,
)

from typing import Union

from tt_lib.utils import _nearest_32
import tt_lib as ttl

import math
import torch
from ttnn._ttnn.operations import pool


class TTPyMaxPool(TTPyOp):
    def __init__(
        self,
        sliding_window_op_params: Union[SlidingWindowOpParams, SlidingWindowOpParamsWithParallelConfig],
        device,
        reader_patterns_cache,
        pad_val=0xF7FF,
        parallel_config_override=None,
        output_mem_config=None,
        deallocate_activation=True,
        act_dtype=None,
        channels=None,
    ):
        if parallel_config_override is None:
            parallel_config_override = {}
        if "max_pool" not in reader_patterns_cache:
            reader_patterns_cache["max_pool"] = {}
        if "halo" not in reader_patterns_cache:
            reader_patterns_cache["halo"] = {}

        for key in reader_patterns_cache:
            assert (
                key == "max_pool" or key == "halo" or key == "conv"
            ), f"reader_patterns_cache should have 1 of the following keys - 'conv', 'max_pool' or 'halo'. Found key - {key}"

        snap_to_tile = parallel_config_override.get("snap_to_tile", False)
        df_needs_tiled = act_dtype is not None and act_dtype == ttl.tensor.DataType.BFLOAT8_B
        conv_parallel_config = determine_parallel_config(
            True,
            0,
            0,
            sliding_window_op_params,
            device,
            config_override=parallel_config_override,
            is_out_tiled=snap_to_tile or df_needs_tiled,
        )
        self.grid_size = (conv_parallel_config.grid_size.x, conv_parallel_config.grid_size.y)
        self.ncores_nhw = conv_parallel_config.num_cores_nhw
        self.shard_grid, self.shard_layout = calculate_shard_grid(self.grid_size, self.ncores_nhw)
        assert (
            self.shard_layout == ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED
        ), "TTPyMaxPool currently only supports height sharding"

        if isinstance(sliding_window_op_params, SlidingWindowOpParams):
            self.sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
                stride_h=sliding_window_op_params.stride_h,
                stride_w=sliding_window_op_params.stride_w,
                pad_h=sliding_window_op_params.pad_h,
                pad_w=sliding_window_op_params.pad_w,
                window_h=sliding_window_op_params.window_h,
                window_w=sliding_window_op_params.window_w,
                batch_size=sliding_window_op_params.batch_size,
                input_h=sliding_window_op_params.input_h,
                input_w=sliding_window_op_params.input_w,
                num_cores_h=self.grid_size[1],
                num_cores_w=self.grid_size[0],
                num_cores_nhw=self.ncores_nhw,
            )
        else:
            self.sliding_window_op_params = sliding_window_op_params

        sliding_window_op_params_hash = get_hash_from_sliding_window_op_params(self.sliding_window_op_params)

        self.device = device

        self.input_sharded_memory_config = calculate_memory_config(
            self.sliding_window_op_params,
            True,
            0 if channels is None else channels,
            calc_input=True,
            tile_size=32 if snap_to_tile else 1,
        )
        self.output_sharded_memory_config = (
            calculate_memory_config(
                self.sliding_window_op_params,
                True,
                0 if channels is None else channels,
                calc_input=False,
                tile_size=32 if snap_to_tile else 1,
            )
            if output_mem_config is None
            else output_mem_config
        )

        self.set_op_configs(
            sliding_window_op_params_hash,
            reader_patterns_cache["max_pool"],
        )
        assert sliding_window_op_params_hash in reader_patterns_cache["max_pool"]
        reader_indices = reader_patterns_cache["max_pool"][sliding_window_op_params_hash]

        self.set_op_weights_biases(
            self.sliding_window_op_params,
            reader_indices,
        )

        self.pad_val = pad_val
        self.untilize_with_halo = TTPyUntilizeWithHalo(
            self.device,
            self.sliding_window_op_params,
            reader_patterns_cache["halo"],
            pad_val=self.pad_val,
            is_out_tiled=snap_to_tile,
        )

        self.deallocate_activation = deallocate_activation

    # override abstract methods from base class TTPyOp
    def set_op_configs(self, sliding_window_op_params_hash, reader_patterns_cache):
        if sliding_window_op_params_hash not in reader_patterns_cache:
            stride_h = self.sliding_window_op_params.stride_h
            stride_w = self.sliding_window_op_params.stride_w
            pad_h = self.sliding_window_op_params.pad_h
            pad_w = self.sliding_window_op_params.pad_w
            window_h = self.sliding_window_op_params.window_h
            window_w = self.sliding_window_op_params.window_w
            batch_size = self.sliding_window_op_params.batch_size
            input_h = self.sliding_window_op_params.input_h
            input_w = self.sliding_window_op_params.input_w
            ncores_h = self.sliding_window_op_params.num_cores_h
            ncores_w = self.sliding_window_op_params.num_cores_w
            ncores_nhw = self.sliding_window_op_params.num_cores_nhw

            input_nchw_shape = [batch_size, 1, input_h, input_w]
            input_shard_height = self.input_sharded_memory_config.shard_spec.shape[0]
            output_shard_height = self.output_sharded_memory_config.shard_spec.shape[0]
            input_padded_width = input_w + 2 * pad_w

            pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
                (1, 1, window_h, window_w, stride_h, stride_w, pad_h, pad_w, 1, 1), input_nchw_shape
            )

            req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
                data_top_left_indices,
                pad_metadata,
                input_padded_width,
                output_shard_height,
                input_shard_height,
                ncores_nhw,
                window_h,
                window_w,
            )

            sliding_window_op_sharded_input_top_left_indices = (
                generate_sliding_window_op_sharded_input_top_left_indices(
                    data_top_left_indices, req_conv_input_shard_start_end, pad_tile=True, pad_last_core=True
                )
            )

            indices_torch_dtype = torch.int16
            indices_tt_dtype = ttl.tensor.DataType.UINT16

            # Create sharded tensor on device for conv_reader_indices
            reader_indices_torch_tensor = torch.tensor(
                [[sliding_window_op_sharded_input_top_left_indices]], dtype=indices_torch_dtype
            )
            reader_indices_tt_tensor = ttl.tensor.Tensor(
                reader_indices_torch_tensor,
                indices_tt_dtype,
            )
            shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
            shard_halo = False
            shard_spec = ttl.tensor.ShardSpec(
                self.shard_grid, [1, reader_indices_tt_tensor.get_legacy_shape()[-1]], shard_orientation, shard_halo
            )
            mem_config = ttl.tensor.MemoryConfig(self.shard_layout, ttl.tensor.BufferType.L1_SMALL, shard_spec)
            reader_indices_sharded_tensor = reader_indices_tt_tensor.to(self.device, mem_config)

            reader_patterns_cache[sliding_window_op_params_hash] = reader_indices_sharded_tensor

        return

    def set_op_weights_biases(self, op_params, reader_indices):
        stride_h = op_params.stride_h
        stride_w = op_params.stride_w
        pad_h = op_params.pad_h
        pad_w = op_params.pad_w
        window_h = op_params.window_h
        window_w = op_params.window_w
        in_n = op_params.batch_size
        in_h = op_params.input_h
        in_w = op_params.input_w

        def max_pool_(activation):
            act_mem_config = activation.memory_config()
            haloed_act = self.untilize_with_halo(activation)

            if self.deallocate_activation:
                activation.deallocate()
            output = pool.max_pool2d_v2(
                haloed_act,
                reader_indices,
                in_n,
                in_h,
                in_w,
                window_h,
                window_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                memory_config=self.output_sharded_memory_config,
            )
            haloed_act.deallocate()
            return output

        self.max_pool = max_pool_

    def __call__(self, activation):
        return self.max_pool(activation)

    def copy_input_to_device(self, input: ttl.tensor.Tensor):
        in_shape = input.get_legacy_shape()
        in_c = in_shape[-1]
        in_n = self.sliding_window_op_params.batch_size
        in_h = self.sliding_window_op_params.input_h
        in_w = self.sliding_window_op_params.input_w
        assert in_c % 16 == 0, "Input channels should be multiple of 16. General case is TODO"
        act_shape = (1, 1, in_n * in_h * in_w, in_c)
        act_reshaped = input.reshape(act_shape)
        padded_nhw = self.input_sharded_memory_config.shard_spec.shape[0] * self.sliding_window_op_params.num_cores_nhw
        if padded_nhw != act_shape[-2]:
            padded_shape = ttl.tensor.Shape(act_shape, (1, 1, padded_nhw, in_c))
            act_reshaped = ttl.tensor.format_input_tensor(
                act_reshaped,
                self.device,
                padded_shape,
                -float("inf"),
                act_reshaped.layout,
            )

        interleaved_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
        )
        mem_config = self.input_sharded_memory_config
        shard_shape = mem_config.shard_spec.shape
        shard_shape[1] = in_c
        mem_config.shard_spec.shape = shard_shape
        act_reshaped = act_reshaped.to(self.device, interleaved_mem_config)
        return ttl.tensor.interleaved_to_sharded(
            act_reshaped,
            mem_config,
            input.get_dtype(),
        )

    def copy_output_from_device(self, output_d: ttl.tensor.Tensor):
        interleaved_mem_config = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
        )
        output_d = ttl.tensor.sharded_to_interleaved(output_d, interleaved_mem_config)
        return output_d.cpu()

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));

    uint32_t arg_index = 0;
    ShardAddrGen<shard_type> addr_gen;
    const uint32_t eth_sender_l1_base_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_sender_l1_sem_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_sender_noc_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t eth_sender_noc_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t shards_per_eth_l1_buffer = get_arg_val<uint32_t>(arg_index++);
    const uint32_t writer_send_sem_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_index++);
    const uint32_t num_input_shards_from_local_ring_index = get_arg_val<uint32_t>(arg_index++);
    const uint32_t half_cb_n_shards = get_arg_val<uint32_t>(arg_index++);

    ShardAddrGen<shard_type>::build_with_placement_new(&addr_gen, arg_index);
    arg_index += addr_gen.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // Used to wait until eth sender has space available
    volatile tt_l1_ptr uint32_t* writer_send_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_send_sem_addr);

    // This is different per writer core
    const uint64_t eth_l1_sender_base_noc_addr =
        get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_base_addr);
    // Used to signal eth sender that data is available. This is different per writer core
    const uint64_t eth_l1_sender_semaphore_addr =
        get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_sem_addr);

    constexpr bool use_optimized = true;
    // one input shard per core for the local chip forward to output tensor
    const uint32_t shard_size = addr_gen.get_shard_size_in_bytes();
    for (uint32_t i = 0; i < num_input_shards_from_local_ring_index; i += shards_per_eth_l1_buffer) {
        uint32_t num_shards_to_send = std::min(shards_per_eth_l1_buffer, num_input_shards_from_local_ring_index - i);
        noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
        noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
        write_and_send_chunk_sharded(cb_id_in0, addr_gen, num_shards_to_send, eth_l1_sender_base_noc_addr, eth_l1_sender_semaphore_addr);
        if (half_cb_n_shards - num_shards_to_send) {
            pop_filler_pages_from_cb(cb_id_in0, half_cb_n_shards - num_shards_to_send);
        }
    }

    // num_transfers = num_devices - 1
    for (uint32_t t = 1; t < num_transfers; ++t) {
        for (uint32_t i = 0; i < num_input_shards_from_local_ring_index; i += shards_per_eth_l1_buffer) {
            uint32_t num_shards_to_send = std::min(shards_per_eth_l1_buffer, num_input_shards_from_local_ring_index - i);
            noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
            noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
            send_chunk_sharded(cb_id_in0, num_shards_to_send, shard_size, eth_l1_sender_base_noc_addr, eth_l1_sender_semaphore_addr);
            if (half_cb_n_shards - num_shards_to_send) {
                pop_filler_pages_from_cb(cb_id_in0, half_cb_n_shards - num_input_shards_from_local_ring_index);
            }
        }
    }
}

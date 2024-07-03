// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

template <typename INTERLEAVED_ADDR_GEN_T>
void stride_for_next_transfer_index(
    bool is_clockwise_direction,
    uint32_t& input_ring_idx,
    uint32_t ring_size,
    uint32_t output_addr_offset,
    uint32_t output_page_offset,
    uint32_t last_output_addr_offset,
    uint32_t last_output_page_offset,

    INTERLEAVED_ADDR_GEN_T &d,
    uint32_t &output_base_page_idx
) {
    if (is_clockwise_direction) {
        if (input_ring_idx == 0) {
            input_ring_idx = ring_size - 1;
            if (output_addr_offset != 0) {
                d.bank_base_address += last_output_addr_offset;
            }
            if (output_page_offset != 0) {
                output_base_page_idx += last_output_page_offset;
            }
        } else {
            input_ring_idx--;
            if (output_addr_offset != 0) {
                d.bank_base_address -= output_addr_offset;
            }
            if (output_page_offset != 0) {
                output_base_page_idx -= output_page_offset;
            }
        }
    } else {
        if (input_ring_idx == ring_size - 1) {//0) {
            input_ring_idx = 0;
            if (output_addr_offset != 0) {
                d.bank_base_address -= last_output_addr_offset;
                // d.bank_base_address = last_output_addr_offset;
            }
            if (output_page_offset != 0) {
                output_base_page_idx -= last_output_page_offset;
                // output_base_page_idx = last_output_page_offset;
            }
        } else {
            input_ring_idx++;
            if (output_addr_offset != 0) {
                d.bank_base_address += output_addr_offset;
            }
            if (output_page_offset != 0) {
                output_base_page_idx += output_page_offset;
            }
        }
    }
}

void kernel_main() {
    //// OLD SEND READER ARGS
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t eth_receiver_l1_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t eth_receiver_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t eth_receiver_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t eth_receiver_l1_semaphore_addr = get_arg_val<uint32_t>(4);
    volatile uint32_t* reader_local_sem_addr_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(5));

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_pages_per_full_chunk = get_compile_time_arg_val(4);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t input_start_idx = get_compile_time_arg_val(6);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(7);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(8);
    constexpr uint32_t row_offset = get_compile_time_arg_val(9);
    constexpr uint32_t col_offset = get_compile_time_arg_val(10);
    constexpr uint32_t num_rows = get_compile_time_arg_val(11);
    constexpr uint32_t num_cols = get_compile_time_arg_val(12);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(13);
    constexpr bool is_clockwise_direction = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(15);
    constexpr uint32_t ring_size = get_compile_time_arg_val(16);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(17);
    static_assert(half_cb_n_pages > rem_num_pages, "half_cb_n_pages must be greater than or equal to rem_num_pages");

    // DPRINT << "RD num_Transfers: " << num_transfers << ", num_full_chunks: " << num_full_chunks << ", rem_num_pages: " << rem_num_pages << "\n";

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    uint64_t eth_semaphore_addr = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_semaphore_addr);
    // std::array<uint64_t, num_buffers_per_channel> eth_semaphore_addrs;
    std::array<uint64_t, num_buffers_per_channel> eth_buffer_addrs;
    uint32_t buffer_size = num_pages_per_full_chunk * page_size;
    for (uint32_t i = 0; i < num_buffers_per_channel; ++i) {
        // eth_semaphore_addrs[i] = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_semaphore_addr + (i * buffer_size));
        eth_buffer_addrs[i] = get_noc_addr(eth_receiver_noc_x, eth_receiver_noc_y, eth_receiver_l1_base_addr + (i * (buffer_size + 16)));
        // DPRINT << "RD buffer_addr[" << i << "]: " << (eth_buffer_addrs[i] & 0xFFFFFFFF) << "\n";
        // DPRINT << "RD sem_addr[" << i << "]: " << (eth_semaphore_addrs[i] & 0xFFFFFFFF) << ", buffer_addr[" << i << "]: " << (eth_buffer_addrs[i] & 0xFFFFFFFF) << "\n";
    }
    uint32_t buffer_index = 0;


    #ifdef RM_INTERLEAVED
    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_size};
    #elif defined TILE_INTERLEAVED
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = page_size,
        .data_format = in0_df
    };
    #endif

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t input_page_idx = input_start_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;

    uint32_t total_num_pages = num_full_chunks * num_pages_per_full_chunk + rem_num_pages;
    uint32_t sem_idx = 1;

    for (uint32_t i = 0; i < total_num_pages; i += num_pages_per_full_chunk) {

        uint32_t num_pages_to_forward = std::min(num_pages_per_full_chunk, total_num_pages - i);
        uint32_t num_filler_pages = num_pages_per_full_chunk - num_pages_to_forward;
        {   // Read from the input tensor

            // DPRINT << "RD rcfit " << num_pages_to_forward << "\n";
            read_chunk_from_input_tensor(input_page_idx, cb_id_in0, s, num_pages_to_forward, page_size);

            if (num_filler_pages != 0) {
                // DPRINT << "RD push_filler_pages_to_cb " << num_pages_to_forward << "\n";
                push_filler_pages_to_cb(cb_id_in0, num_filler_pages);
            }
        }

        // num_transfers = num_devices - 1
        for (uint32_t i = 0; i < num_transfers; ++i) {

            DPRINT << "RD nsw \n";
            noc_semaphore_wait(reader_local_sem_addr_ptr, 1);
            noc_semaphore_set(reader_local_sem_addr_ptr, 0);
            // DPRINT << "RD fc \n";
            DPRINT << "RD buf idx " << buffer_index << "\n";
            fetch_chunk(cb_id_in0, num_pages_to_forward, page_size, eth_buffer_addrs[buffer_index]);
            noc_semaphore_inc(eth_semaphore_addr, 1);
            // noc_semaphore_inc(eth_semaphore_addrs[buffer_index], 1);

            if (num_filler_pages != 0) {
                // DPRINT << "RD filler pages \n";
                push_filler_pages_to_cb(cb_id_in0, num_filler_pages);
            }

            buffer_index = (buffer_index == num_buffers_per_channel - 1) ? 0 : buffer_index + 1;
        }
    }


    DPRINT << "RD DONE \n";
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    // Different per worker receiver writer
    const uint32_t edm_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t edm_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t edm_sem_addr = get_arg_val<uint32_t>(3);
    const uint32_t edm_base_buffer_address = get_arg_val<uint32_t>(4);
    volatile uint32_t* local_sem_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(5));

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_pages_per_full_chunk = get_compile_time_arg_val(4);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t output_start_page_idx = get_compile_time_arg_val(6);
    constexpr uint32_t output_start_addr_offset = get_compile_time_arg_val(7);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(8);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(9);
    constexpr uint32_t row_offset = get_compile_time_arg_val(10);
    constexpr uint32_t col_offset = get_compile_time_arg_val(11);
    constexpr uint32_t num_rows = get_compile_time_arg_val(12);
    constexpr uint32_t num_cols = get_compile_time_arg_val(13);
    constexpr uint32_t last_output_page_offset = get_compile_time_arg_val(14);
    constexpr uint32_t output_page_offset = get_compile_time_arg_val(15);
    constexpr uint32_t last_output_addr_offset = get_compile_time_arg_val(16);
    constexpr uint32_t output_addr_offset = get_compile_time_arg_val(17);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(18);
    // Same per worker receiver writer
    constexpr bool is_clockwise_direction = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(20);
    constexpr uint32_t ring_size = get_compile_time_arg_val(21);
    static_assert(half_cb_n_pages > rem_num_pages, "half_cb_n_pages must be greater than or equal to rem_num_pages");

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    #ifdef RM_INTERLEAVED
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
    #elif defined TILE_INTERLEAVED
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = output_page_size,
        .data_format = in0_df
    };
    #endif

    // Each worker receiver writer matches with a specific worker sender reader
    // Used to signal that data has been committed to memory and can be read
    const uint64_t edm_semaphore_noc_addr = get_noc_addr(edm_noc_x, edm_noc_y, edm_sem_addr);
    const uint64_t edm_l1_sender_base_noc_addr = get_noc_addr(edm_noc_x, edm_noc_y, edm_base_buffer_address);

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t output_base_page_idx = output_start_page_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;

    uint32_t total_num_pages = num_full_chunks * num_pages_per_full_chunk + rem_num_pages;

    for (uint32_t i = 0; i < total_num_pages; i += num_pages_per_full_chunk) {
        uint32_t num_pages_to_forward = std::min(num_pages_per_full_chunk, total_num_pages - i);
        uint32_t num_filler_pages = num_pages_per_full_chunk - num_pages_to_forward;

        for (uint32_t i = 0; i < num_transfers; ++i) {
            noc_semaphore_wait(local_sem_ptr, 1);
            noc_semaphore_set(local_sem_ptr, 0);
            write_and_send_chunk(
                output_page_idx,
                col_idx,
                row_idx,
                cb_id,
                d,
                num_cols,
                num_rows,
                col_offset,
                row_offset,
                num_pages_to_forward,
                page_size,
                edm_l1_sender_base_noc_addr,
                edm_semaphore_noc_addr);

            if (num_filler_pages != 0) {
                pop_filler_pages_from_cb(cb_id, num_filler_pages);
            }

            if (is_clockwise_direction) {
                if (input_ring_idx == 0) {
                    input_ring_idx = ring_size - 1;
                    if constexpr(output_addr_offset != 0) {
                        d.bank_base_address += last_output_addr_offset;
                    }
                    if constexpr(output_page_offset != 0) {
                        output_base_page_idx += last_output_page_offset;
                    }
                } else {
                    input_ring_idx--;
                    if constexpr(output_addr_offset != 0) {
                        d.bank_base_address -= output_addr_offset;
                    }
                    if constexpr(output_page_offset != 0) {
                        output_base_page_idx -= output_page_offset;
                    }
                }
            } else {
                if (input_ring_idx == ring_size - 1) {
                    input_ring_idx = 0;
                    if constexpr(output_addr_offset != 0) {
                        d.bank_base_address -= last_output_addr_offset;
                    }
                    if constexpr(output_page_offset != 0) {
                        output_base_page_idx -= last_output_page_offset;
                    }
                } else {
                    input_ring_idx++;
                    if constexpr(output_addr_offset != 0) {
                        d.bank_base_address += output_addr_offset;
                    }
                    if constexpr(output_page_offset != 0) {
                        output_base_page_idx += output_page_offset;
                    }
                }

            }
            output_page_idx = output_base_page_idx;
            col_idx = col_start_idx;
            row_idx = row_start_idx;
        }
    }
}

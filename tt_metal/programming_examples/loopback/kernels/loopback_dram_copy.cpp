// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "Hello from the kernel" << ENDL();
    std::uint32_t l1_buffer_addr        = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t dram_src_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(4);
    std::uint32_t dram_dst_noc_x        = get_arg_val<uint32_t>(5);
    std::uint32_t dram_dst_noc_y        = get_arg_val<uint32_t>(6);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(7);

    DPRINT << "Printing some info:" << ENDL();
    DPRINT << "l1_buffer_addr       = 0x" 	<< HEX() << l1_buffer_addr << ENDL();
    DPRINT << "dram_buffer_src_addr = 0x" << HEX() << dram_buffer_src_addr << ENDL();
    DPRINT << "dram_src_noc_x       = 0x" 	<< HEX() << dram_src_noc_x << ENDL();
    DPRINT << "dram_src_noc_y       = 0x" 	<< HEX() << dram_src_noc_y << ENDL();
    DPRINT << "dram_buffer_dst_addr = 0x" << HEX() << dram_buffer_dst_addr << ENDL();
    DPRINT << "dram_dst_noc_x       = 0x" 	<< HEX() << dram_dst_noc_x << ENDL();
    DPRINT << "dram_dst_noc_y       = 0x" 	<< HEX() << dram_dst_noc_y << ENDL();
    DPRINT << "dram_buffer_size     = 0x" 	<< HEX() << dram_buffer_size << ENDL();


    std::uint32_t cur_pc = 0;
    std::uint32_t cur_sp = 0;

    __asm volatile(
        "auipc %0, 0\n    \
	 addi %1, sp, 0\n"
	 : "=r"(cur_pc), "=r"(cur_sp)
	 ::
    );

    DPRINT << "cur_pc               = 0x" << HEX() << cur_pc << ENDL();
    DPRINT << "cur_sp               = 0x" << HEX() << cur_sp << ENDL();

    noc_async_write_barrier();

    std::uint64_t dram_buffer_src_noc_addr = get_noc_addr(dram_src_noc_x, dram_src_noc_y, dram_buffer_src_addr);
    DPRINT << "dram_buffer_src_noc_addr    = 0x" << HEX() << dram_buffer_src_noc_addr << ENDL();
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
    noc_async_read_barrier();


    DPRINT << "Printing L1 buffer contents post noc_async_read()" << ENDL();
    
    std::uint32_t *l1_buf = (std::uint32_t *) l1_buffer_addr;
    for(std::uint32_t i = 0; i < dram_buffer_size/4; i++){
        DPRINT << "[0x" << HEX() << i*4 << "] 0x" << HEX() << l1_buf[i] << ENDL();
    }

    std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr(dram_dst_noc_x, dram_dst_noc_y, dram_buffer_dst_addr);
    DPRINT << "dram_buffer_dst_noc_addr    = 0x" << HEX() << dram_buffer_dst_noc_addr << ENDL();
    noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
    noc_async_write_barrier();
    DPRINT << "Kernel made it through" << ENDL();
}

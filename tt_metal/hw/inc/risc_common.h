// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _RISC_COMMON_H_
#define _RISC_COMMON_H_

#include <stdint.h>

#include <cstdint>

#include "eth_l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "limits.h"
#include "mod_div_lib.h"
#include "noc_overlay_parameters.h"
#include "noc_parameters.h"
#include "stream_io_map.h"
#include "tensix.h"

#define NOC_X(x) NOC_0_X(noc_index, noc_size_x, (x))
#define NOC_Y(y) NOC_0_Y(noc_index, noc_size_y, (y))

#define TILE_WORD_2_BIT ((256 + 64 + 32) >> 4)
#define TILE_WORD_4_BIT ((512 + 64 + 32) >> 4)
#define TILE_WORD_8_BIT ((32 * 32 * 1 + 64 + 32) >> 4)
#define TILE_WORD_16_BIT ((32 * 32 * 2 + 32) >> 4)
#define TILE_WORD_32_BIT ((32 * 32 * 4 + 32) >> 4)

#ifdef COMPILE_FOR_BRISC
constexpr std::uint32_t L1_ARG_BASE = BRISC_L1_ARG_BASE;
constexpr std::uint32_t L1_RESULT_BASE = BRISC_L1_RESULT_BASE;
#elif defined(COMPILE_FOR_NCRISC)
constexpr std::uint32_t L1_ARG_BASE = NCRISC_L1_ARG_BASE;
constexpr std::uint32_t L1_RESULT_BASE = NCRISC_L1_RESULT_BASE;
#elif defined(COMPILE_FOR_TRISC)
constexpr std::uint32_t L1_ARG_BASE = TRISC_L1_ARG_BASE;
constexpr std::uint32_t L1_RESULT_BASE = TRISC_L1_ARG_BASE + 1024;
#elif defined(COMPILE_FOR_ERISC)
constexpr std::uint32_t L1_ARG_BASE = eth_l1_mem::address_map::ERISC_L1_ARG_BASE;
constexpr std::uint32_t L1_RESULT_BASE = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
#elif defined(COMPILE_FOR_IDLE_ERISC)
constexpr std::uint32_t L1_ARG_BASE = IDLE_ERISC_L1_ARG_BASE;
constexpr std::uint32_t L1_RESULT_BASE = IDLE_ERISC_L1_RESULT_BASE;
#endif

const uint32_t STREAM_RESTART_CHECK_MASK = (0x1 << 3) - 1;

const uint32_t MAX_TILES_PER_PHASE = 2048;

extern uint8_t my_x[NUM_NOCS];
extern uint8_t my_y[NUM_NOCS];

inline void WRITE_REG(uint32_t addr, uint32_t val) {
    volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
    ptr[0] = val;
}

inline uint32_t READ_REG(uint32_t addr) {
    volatile tt_reg_ptr uint32_t* ptr = (volatile tt_reg_ptr uint32_t*)addr;
    return ptr[0];
}

inline uint32_t dram_io_incr_ptr(uint32_t curr_ptr, uint32_t incr, uint32_t buf_size_q_slots) {
    uint32_t next_ptr = curr_ptr + incr;
    uint32_t double_buf_size_q_slots = 2 * buf_size_q_slots;
    if (next_ptr >= double_buf_size_q_slots) {
        next_ptr -= double_buf_size_q_slots;
    }
    return next_ptr;
}

inline __attribute__((always_inline)) uint32_t dram_io_empty(uint32_t rd_ptr, uint32_t wr_ptr) {
    return (rd_ptr == wr_ptr);
}

inline __attribute__((always_inline)) uint32_t
dram_io_local_empty(uint32_t local_rd_ptr, uint32_t rd_ptr, uint32_t wr_ptr) {
    if (rd_ptr == wr_ptr)
        return true;

    uint32_t case1 = rd_ptr < wr_ptr && (local_rd_ptr < rd_ptr || local_rd_ptr >= wr_ptr);
    uint32_t case2 = rd_ptr > wr_ptr && wr_ptr <= local_rd_ptr && local_rd_ptr < rd_ptr;

    return case1 || case2;
}

inline uint32_t dram_io_full(uint32_t rd_ptr, uint32_t wr_ptr, uint32_t buf_size_q_slots) {
    uint32_t wr_ptr_reduced_by_q_slots = wr_ptr - buf_size_q_slots;
    uint32_t rd_ptr_reduced_by_q_slots = rd_ptr - buf_size_q_slots;
    uint32_t case1 = (wr_ptr_reduced_by_q_slots == rd_ptr);
    uint32_t case2 = (rd_ptr_reduced_by_q_slots == wr_ptr);
    return case1 || case2;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_inc_wrap(uint32_t buf_ptr, uint32_t inc, uint32_t buf_size) {
    uint32_t result = buf_ptr + inc;
    if (result >= buf_size) {
        result -= buf_size;
    }
    return result;
}

inline __attribute__((always_inline)) uint32_t buf_ptr_dec_wrap(uint32_t buf_ptr, uint32_t dec, uint32_t buf_size) {
    uint32_t result = buf_ptr;
    if (dec > result) {
        result += buf_size;
    }
    result -= dec;
    return result;
}

inline __attribute__((always_inline)) uint32_t reg_read(uint32_t addr) {
    volatile tt_reg_ptr uint32_t* p_reg = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(addr);
    return p_reg[0];
}

inline void assert_trisc_reset() {
    uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
    uint32_t trisc_reset_mask = RISCV_SOFT_RESET_0_TRISCS;
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 | trisc_reset_mask);
}

inline void deassert_trisc_reset() {
    uint32_t soft_reset_0 = READ_REG(RISCV_DEBUG_REG_SOFT_RESET_0);
    uint32_t trisc_reset_mask = RISCV_SOFT_RESET_0_TRISCS;
    WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset_0 & ~trisc_reset_mask);
}

inline void deassert_all_reset() { WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NONE); }

inline void assert_just_ncrisc_reset() { WRITE_REG(RISCV_DEBUG_REG_SOFT_RESET_0, RISCV_SOFT_RESET_0_NCRISC); }

inline uint32_t special_mult(uint32_t a, uint32_t special_b) {
    if (special_b == TILE_WORD_8_BIT)
        return a * TILE_WORD_8_BIT;
    else if (special_b == TILE_WORD_16_BIT)
        return a * TILE_WORD_16_BIT;
    else if (special_b == TILE_WORD_4_BIT)
        return a * TILE_WORD_4_BIT;
    else if (special_b == TILE_WORD_2_BIT)
        return a * TILE_WORD_2_BIT;
    else if (special_b == TILE_WORD_32_BIT)
        return a * TILE_WORD_32_BIT;

    while (true);
    return 0;
}

// risc_init function isn't required for TRISCS
#if !defined(COMPILE_FOR_TRISC)  // BRISC, NCRISC, ERISC, IERISC
#include "noc_nonblocking_api.h"

#if defined(COMPILE_FOR_ERISC)
// ERISC needs to place this function in a specific section
__attribute__((section("code_l1")))
#endif  // defined(COMPILE_FOR_ERISC)
void risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}
#endif  // !defined(COMPILE_FOR_TRISC)

inline void breakpoint_(uint32_t line) {
    /*
        When called, writes the stack pointer to a known location
        in memory (unique for each core) and then hangs until the
        user explicitly continues
    */
    uint32_t BREAKPOINT;
    uint32_t LNUM;
    volatile tt_l1_ptr uint32_t* bp;
    volatile tt_l1_ptr uint32_t* lnum;

#define MACRO_SP_AUX(SP) #SP
#define MACRO_SP(SP) MACRO_SP_AUX(SP)

// Need to use macros for inline assembly in order to create a string literal
#if defined(COMPILE_FOR_NCRISC)
    asm("li t0, " MACRO_SP(NCRISC_SP_MACRO));
    BREAKPOINT = NCRISC_BREAKPOINT;
    LNUM = NCRISC_BP_LNUM;
#elif defined(COMPILE_FOR_BRISC)
    asm("li t0, " MACRO_SP(BRISC_SP_MACRO));
    BREAKPOINT = BRISC_BREAKPOINT;
    LNUM = BRISC_BP_LNUM;
#elif COMPILE_FOR_TRISC == 0
    asm("li t0, " MACRO_SP(TRISC0_SP_MACRO));
    BREAKPOINT = TRISC0_BREAKPOINT;
    LNUM = TRISC0_BP_LNUM;
#elif COMPILE_FOR_TRISC == 1
    asm("li t0, " MACRO_SP(TRISC1_SP_MACRO));
    BREAKPOINT = TRISC1_BREAKPOINT;
    LNUM = TRISC1_BP_LNUM;
#elif COMPILE_FOR_TRISC == 2
    asm("li t0, " MACRO_SP(TRISC2_SP_MACRO));
    BREAKPOINT = TRISC2_BREAKPOINT;
    LNUM = TRISC2_BP_LNUM;
#endif

    // Write '1' to breakpoint location so that this core keeps
    // busy looping until host releases it
    asm("sw sp, 0(t0)");
    bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(BREAKPOINT);
    bp[0] = 1;

    lnum = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(LNUM);
    lnum[0] = line;

    while (bp[0] == 1);
}

#define breakpoint() breakpoint_(__LINE__);

// Helper function to wait for a specified number of cycles, safe to call in erisc kernels.
#if defined(COMPILE_FOR_ERISC)
#include "erisc.h"
#endif
inline void riscv_wait(uint32_t cycles) {
    volatile uint tt_reg_ptr* clock_lo = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint tt_reg_ptr* clock_hi = reinterpret_cast<volatile uint tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    uint64_t wall_clock_timestamp = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    uint64_t wall_clock = 0;
    do {
#if defined(COMPILE_FOR_ERISC)
        internal_::risc_context_switch();
#endif
        wall_clock = clock_lo[0] | ((uint64_t)clock_hi[0] << 32);
    } while (wall_clock < (wall_clock_timestamp + cycles));
}

// Disables Blackhole's L1 cache. Grayskull and Wormhole do not have L1 cache
inline __attribute__((always_inline)) void disable_lowcache() {
#ifdef ARCH_BLACKHOLE
    // asm(R"ASM(
    //         csrrsi zero, 0x7c0, 0x8
    //       )ASM");
    asm(R"ASM(
        .option push
        li   t1, 0x1
        slli t1, t1, 3
        csrrs zero, 0x7c0, t1
        .option pop
         )ASM" ::
            : "t1");
#endif
}

#endif

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "chlkc_list.h"
#include "ckernel.h"
#include "ckernel_debug.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "hostdevcommon/kernel_structs.h"
#include "risc_attribs.h"

#define ALWI inline __attribute__((always_inline))

#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#include "llk_math_common_api.h"
#include "llk_math_matmul_api.h"
#include "llk_math_reduce_api.h"
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_unary_sfpu_api.h"
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_io_pack.h"
#include "llk_pack_api.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_A_api.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_reduce_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool fast_and_approx = true>
ALWI void rsqrt_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_rsqrt_init<fast_and_approx>()));
}

/**
 * Performs element-wise computation of reciprocal sqrt on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | fast_and_approx | Computation to be done faster and
 * approximate                              | bool     |                                                       | False |
 */
template <bool fast_and_approx = true>
ALWI void rsqrt_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_rsqrt<fast_and_approx>(idst)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sigmoid_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_sigmoid_init<APPROX>()));  // TODO(AP): move out init
}

/**
 * Performs element-wise computation of sigmoid on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void sigmoid_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_sigmoid<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void log_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_init<APPROX>()));  // TODO(AP): move out init
}

/**
 * Performs element-wise computation of logarithm on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void log_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_log<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void log_with_base_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base_init<APPROX>()));  // TODO(AP): move out init
}

/**
 * Performs element-wise computation of logarithm with a specified base on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | base_scale      | The log base | uint32_t |  Postive
 * integers                                     | True     |
 */
ALWI void log_with_base_tile(uint32_t idst, uint32_t base_scale) {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base<APPROX>(idst, base_scale)));
}

// TODO: Move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void tanh_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_tanh_init<APPROX>()));  // TODO(AP): move out init
}

// TODO: Move to trigonometry.h
/**
 * Performs element-wise computation of tanh on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void tanh_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_tanh<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void signbit_tile_init() { MATH((llk_math_eltwise_unary_sfpu_signbit_init<APPROX>())); }

/**
 * Sets the sign bit of each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to modify the sign bit of     | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void signbit_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_signbit<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void floor_tile_init() { MATH((llk_math_eltwise_unary_sfpu_floor_init<APPROX>())); }

/**
 * Performs floor operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to modify the sign bit of     | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void floor_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_floor<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void trunc_tile_init() { MATH((llk_math_eltwise_unary_sfpu_trunc_init<APPROX>())); }

/**
 * Performs trunc operation on each row of a tile.
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to modify the sign bit of     | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void trunc_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_trunc<APPROX>(idst))); }

/**
 * Performs element-wise computation of absolute value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void abs_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_abs<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void abs_tile_init() { MATH((llk_math_eltwise_unary_sfpu_abs_init<APPROX>())); }

/**
 * Will store in the output of the compute core the signum of the tile.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void sign_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_sign<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void sign_tile_init() { MATH((llk_math_eltwise_unary_sfpu_sign_init<APPROX>())); }

/**
 * Performs element-wise computation of square value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void square_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_square<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void square_tile_init() { MATH((llk_math_eltwise_unary_sfpu_square_init<APPROX>())); }

// compare to zero operators

/**
 * Will store in the output of the compute core True if each element of a tile is less than zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */

ALWI void ltz_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_ltz<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void ltz_tile_init() { MATH((llk_math_eltwise_unary_sfpu_ltz_init<APPROX>())); }

/**
 * Will store in the output of the compute core True if each element of a equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */

ALWI void eqz_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_eqz<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void eqz_tile_init() { MATH((llk_math_eltwise_unary_sfpu_eqz_init<APPROX>())); }

/**
 * Will store in the output of the compute core True if each element is less than or equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */

ALWI void lez_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_lez<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void lez_tile_init() { MATH((llk_math_eltwise_unary_sfpu_lez_init<APPROX>())); }

/**
 * Performs element-wise multiplication on each row of a tile.
 * The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void tiled_prod_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_tiled_prod<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void tiled_prod_tile_init() { MATH((llk_math_eltwise_unary_sfpu_tiled_prod_init<APPROX>())); }

/**
 * Will store in the output of the compute core True if each element is greater than zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */

ALWI void gtz_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_gtz<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gtz_tile_init() { MATH((llk_math_eltwise_unary_sfpu_gtz_init<APPROX>())); }

/**
 * Will store in the output of the compute core True if each element is not equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void nez_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_nez<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void nez_tile_init() { MATH((llk_math_eltwise_unary_sfpu_nez_init<APPROX>())); }

/**
 * Will store in the output of the compute core True if each element is greater than or equal to zero.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void gez_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_gez<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void gez_tile_init() { MATH((llk_math_eltwise_unary_sfpu_gez_init<APPROX>())); }

// POWER : y = x^(const param0)
/**
 * Performs element-wise computation of power operation (x ^(const param0)) value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value of the exponent in the power
 * operation                           | uint32_t |                                                       | True     |
 */
ALWI void power_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_power<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void power_tile_init() { MATH((llk_math_eltwise_unary_sfpu_power_init<APPROX>())); }

// MAX : y = max(idst0, idst1)
/**
 * Performs element-wise computation of max value on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * TODO: fix idst1.
 * currently idst1 is not used and (idst0 + 1) is used for max.
 * because don't know how to use 2 dst register with sfpu.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | idst1           | The index of the tile in DST register
 * buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void max_tile(uint32_t idst0, uint32_t idst1) { MATH((llk_math_eltwise_unary_sfpu_max<APPROX>(idst0))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void max_tile_init() { MATH((llk_math_eltwise_unary_sfpu_max_init<APPROX>())); }

// exp2 : y = 2 ^ x  ==> [y = exp(x * log(2))]
/**
 * Performs element-wise computation of 2^x value where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void exp2_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_exp2<true>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void exp2_tile_init() { MATH((llk_math_eltwise_unary_sfpu_exp2_init<true>())); }

// heaviside : y = 0 if x < 0 , 1 if x > 0 , else value
/**
 * Performs element-wise computation of:  y = 0 if x < 0 , 1 if x > 0 , y= value  where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value the output is if the input
 * is greater than 0                     | uint32_t |                                                       | True     |
 */
ALWI void heaviside_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_heaviside<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void heaviside_tile_init() { MATH((llk_math_eltwise_unary_sfpu_heaviside_init<APPROX>())); }

/**
 * Performs element-wise left_shift computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value the output is if the input
 * is greater than 0                     | uint32_t |                                                       | True     |
 */
ALWI void left_shift_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_left_shift<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void left_shift_tile_init() { MATH((llk_math_eltwise_unary_sfpu_left_shift_init<APPROX>())); }

/**
 * Performs element-wise right_shift computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value the output is if the input
 * is greater than 0                     | uint32_t |                                                       | True     |
 */
ALWI void right_shift_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_right_shift<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void right_shift_tile_init() { MATH((llk_math_eltwise_unary_sfpu_right_shift_init<APPROX>())); }

/**
 * Performs element-wise mod computation on input x/y, where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value the output is if the input
 * is greater than 0                     | uint32_t |                                                       | True     |
 */
ALWI void mod_tile(uint32_t idst, uint32_t param0) { MATH((llk_math_eltwise_unary_sfpu_mod<APPROX>(idst, param0))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void mod_tile_init() { MATH((llk_math_eltwise_unary_sfpu_mod_init<APPROX>())); }

// unary ne : if x !=value --> 1.0, else 0.0
/**
 * Performs element-wise computation of:  result = 1 if x!=value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value to be compared with the
 * input tensor                             | uint32_t |                                                       | True |
 */
ALWI void unary_ne_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_ne<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_ne_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_ne_init<APPROX>())); }

// expm1 : (exp(x) - 1)
/**
 * Performs element-wise computation of exp(x) - 1, v where x is each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void expm1_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_expm1<true>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void expm1_tile_init() { MATH((llk_math_eltwise_unary_sfpu_expm1_init<true>())); }

// TODO: move to trigonometry.h
/**
 * Performs element-wise computation of arcsine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void asin_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_asin<true>(idst))); }

// TODO: move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void asin_tile_init() { MATH((llk_math_eltwise_unary_sfpu_asin_init<true>())); }

// TODO: move to trigonometry.h
/**
 * Performs element-wise computation of arctan on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void atan_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_atan<true>(idst))); }

// TODO: move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void atan_tile_init() { MATH((llk_math_eltwise_unary_sfpu_atan_init<true>())); }

// TODO: move to trigonometry.h
/**
 * Performs element-wise computation of arccossine on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void acos_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_acos<true>(idst))); }

// TODO: move to trigonometry.h
/**
 * Please refer to documentation for any_init.
 */
ALWI void acos_tile_init() { MATH((llk_math_eltwise_unary_sfpu_acos_init<true>())); }

// silu
// Function SILU (same as Swish)
// use activation Silu[x] = x*Sigmoid[x]
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html?highlight=silu#torch.nn.SiLU
ALWI void silu_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_silu<APPROX>(idst))); }

ALWI void silu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_silu_init<APPROX>())); }

// topK local sort
/**
 * Performs local sort stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The algorithm used to implement TopK is found here:
 * https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
 *
 * The local sort stage sorts the data into length-K subsequences of
 * alternating directions, in place. If i_start_phase != i_end_phase, all
 * phases in the range i_start_phase to i_end_phase (inclusive) are computed.
 * If i_start_phase == i_end_phase, only that phase is computed, with
 * i_start_step and i_end_step defining how many steps are computed. This can
 * be used to extend the OP support for cases when K > 64.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that local sort is done across columns on 64 values spanning across
 * 2 tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | idir            | The sorting direction of the local
 * sort (0 == decreasing, 1 == increasing) | int32    | 0 to 1                                                | True |
 * | i_end_phase     | The end phase of the local sort (should be set to log(K)-1)                | int32    | 1 to 5 |
 * True     | | i_start_phase   | The start phase of the local sort (should be set to 0)                     | int32 | 0
 * to 5                                                | False    | | i_end_step      | The end step to perform if
 * i_start_phase == i_end_phase                    | int32    | 4 to 6                                                |
 * False    | | i_start_step    | The start step to perform if i_start_phase == i_end_phase                  | int32 | 4
 * to 6                                                | False    |
 */
ALWI void topk_local_sort(
    uint32_t idst, int idir, int i_end_phase, int i_start_phase = 0, int i_end_step = 0, int i_start_step = 0) {
    MATH((llk_math_eltwise_unary_sfpu_topk_local_sort<true>(
        idst, idir, i_end_phase, i_start_phase, i_end_step, i_start_step)));
}

// topK merge
/**
 * Performs merge stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The merge stage combines length-K subsequences that are 2^m_iter apart,
 * such that the first subsequence receives the top K values, and the
 * second subsequence receives the bottom K values.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that merge is done across columns on values spanning across 2
 * tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | m_iter          | The index of the merge & rebuild
 * iteration of the algorithm                | int32    | 0 to 9                                                | True |
 * | k               | The number of sorted values to return                                      | int32    | {4, 8,
 * 16, 32, 64}                                    | True     |
 */
ALWI void topk_merge(uint32_t idst, int m_iter, int k) {
    MATH((llk_math_eltwise_unary_sfpu_topk_merge<true>(idst, m_iter, k)));
}

// topK rebuild
/**
 * Performs rebuild stage of TopK algorithm on the two data tiles and two
 * index tiles that are pre-loaded in DST register. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * The rebuild stage sorts the length-K subsequences that are 2^(m_iter+1)
 * apart, such that they alternate directions.
 *
 * Note that the two data tiles need to be loaded into the DST register
 * before the invocation of this call. The corresponding index tiles should
 * also be loaded in with the data tiles, at a DST offset of 2 tiles.
 *
 * Note that rebuild is done across columns on values spanning across 2
 * tiles.
 *
 * Note: idist should be set to 0
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | idir            | The sorting direction of the local
 * sort (0 == decreasing, 1 == increasing) | bool     | 0 to 1                                                | True |
 * | m_iter          | The index of the merge & rebuild iteration of the algorithm                | int32    | 0 to 9 |
 * True     | | k               | The number of sorted values to return                                      | int32 |
 * {4, 8, 16, 32, 64}                                    | True     | | logk            | The log of K | int32    | 2 to
 * 6                                                | True     | | skip_second     | Whether or not to skip second tile
 * | int32    | 0 to 1                                                | True     |
 */
ALWI void topk_rebuild(uint32_t idst, bool idir, int m_iter, int k, int logk, int skip_second) {
    MATH((llk_math_eltwise_unary_sfpu_topk_rebuild<true>(idst, idir, m_iter, k, logk, skip_second)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void topk_tile_init() { MATH((llk_math_eltwise_unary_sfpu_topk_init<true>())); }

/**
 * Pauses the cores so that the debug interface can be used to inspect the value of the registers.
 *
 * Return value: None
 */
ALWI void dbg_halt() {
    PACK(dbg_thread_halt<PackThreadId>());
    UNPACK(dbg_thread_halt<UnpackThreadId>());
    MATH(dbg_thread_halt<MathThreadId>());
}

/**
 * Resumes the execution of the cores after a debug halt.
 *
 * Return value: None
 */
ALWI void dbg_unhalt() {
    PACK(dbg_thread_unhalt<PackThreadId>());
    UNPACK(dbg_thread_unhalt<UnpackThreadId>());
    MATH(dbg_thread_unhalt<MathThreadId>());
}

/**
 * Reads the contents of the specified row of the destination register. It reads 8 dwords at a time.
 *
 * | Argument        | Description                                                                | Type      | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|-----------|-------------------------------------------------------|----------|
 * | row_addr        | The row address in the destination register to read                        | int       | | True |
 * | rd_data         | The array of 8 dwords to store the data                                    | uint32_t* | | True |
 *
 * Return value: None
 */
ALWI void dbg_read_dest_acc_row(int row_addr, uint32_t *rd_data) {
    MATH((dbg_get_array_row(dbg_array_id::DEST, row_addr, rd_data)));
}

// unary gt : if x > value --> 1.0, else 0.0
/**
 * Performs element-wise computation of:  result = 1 if x > value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value to be compared with the
 * input tensor                             | uint32_t |                                                       | True |
 */
ALWI void unary_gt_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_gt<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_gt_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_gt_init<APPROX>())); }

// unary lt : if x < value --> 1.0, else 0.0
/**
 * Performs element-wise computation of:  result = 1 if x < value , where x is each element of a tile
 * in DST register at index tile_index. The value is provided as const param0 The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be
 * less than the size of the DST register buffer | True     | | param0          | The value to be compared with the
 * input tensor                             | uint32_t |                                                       | True |
 */
ALWI void unary_lt_tile(uint32_t idst, uint32_t param0) {
    MATH((llk_math_eltwise_unary_sfpu_unary_lt<APPROX>(idst, param0)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void unary_lt_tile_init() { MATH((llk_math_eltwise_unary_sfpu_unary_lt_init<APPROX>())); }

}  // namespace ckernel

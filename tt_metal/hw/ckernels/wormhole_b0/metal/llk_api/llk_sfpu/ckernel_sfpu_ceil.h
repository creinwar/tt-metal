// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "noc_nonblocking_api.h"
#include "limits.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_ceil()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat input = dst_reg[0];
        vFloat result;

        v_if (input <= SHRT_MIN || input > SHRT_MAX) {
            result = input;
        }
        v_endif;

        v_if (input > SHRT_MIN && input <= SHRT_MAX) {
            vInt tmp = float_to_int16(input); //TODO: Replace float_to_int16 to float_to_int32 once it is available
            result = int32_to_float(tmp);
        }
        v_endif;

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel

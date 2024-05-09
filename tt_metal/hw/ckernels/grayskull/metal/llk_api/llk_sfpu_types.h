// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum SfpuType {
    tanh,
    hardtanh,
    gelu,
    exponential,
    exp_with_base,
    sigmoid,
    sigmoid_appx,
    reciprocal,
    sqrt,
    rsqrt,
    lrelu,
    power,
    square,
    tanh_derivative,
    log,
    log_with_base,
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    less_than_equal_zero,
    greater_than_zero,
    clamp,
    gelu_derivative,
    dropout,
    abs,
    sign,
    max,
    min,
    sine,
    cosine,
    tan,
    relu_min,
    relu_max,
    elu,
    exp2,
    heaviside,
    expm1,
    signbit,
    asin,
    acos,
    atan,
    erf,
    erfc,
    isfinite,
    isinf,
    isposinf,
    isneginf,
    isnan,
    logical_not_unary,
    erfinv,
    i0,
    silu,
    mask,
    negative,
    unary_ne,
    unary_gt,
    unary_lt,
    tiled_prod,
    left_shift,
    right_shift,
    unary_floor,
    mod,
    trunc,
    unused,
};

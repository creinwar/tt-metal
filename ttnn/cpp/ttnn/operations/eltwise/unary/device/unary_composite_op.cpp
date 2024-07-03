// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <iostream>

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::unary{

enum class UnaryCompositeOpType {
    ACOSH,
    ASINH,
    ATANH,
    CBRT,
    COSH,
    DIGAMMA,
    HARDSWISH,
    HARDSIGMOID,
    HARDTANH,
    LGAMMA,
    LOG1P,
    MISH,
    MULTIGAMMALN,
    SINH,
    SOFTSIGN,
    SWISH,
    TANHSHRINK,
    TRIL,
    TRIU,
};


namespace utils {

// acosh(x) = log(x + sqrt(x^2 - 1))
// Tensor _acosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor t_one = ones_like(input_a, output_mem_config);
//     Tensor t_result(input_a);
//     {
//         Tensor ln_res(input_a);
//         {
//             Tensor x_abs = abs(input_a, output_mem_config);
//             Tensor x_sq_m1(input_a);
//             {
//                 Tensor x_sq = ttnn::square(x_abs, output_mem_config);
//                 x_sq_m1 = sub_unary(x_sq, 1.0f, output_mem_config);
//             }
//             ln_res = log(
//                 ttnn::add(x_abs, sqrt(x_sq_m1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
//         }
//         // To handle inputs <= 1
//         // input < 1, output is nan
//         // input > 1, output is acosh(input)
//         Tensor scalar = ttnn::operations::creation::create_scalar(
//             std::nanf(""), input_a.get_dtype(), Layout::TILE, input_a.device());
//         Tensor nan_res = ttnn::multiply(
//             ttnn::le(input_a, t_one, std::nullopt, output_mem_config), scalar, std::nullopt, output_mem_config);
//         scalar.deallocate();
//         t_result = ttnn::multiply(
//             ttnn::gt(input_a, t_one, std::nullopt, output_mem_config), ln_res, std::nullopt, output_mem_config);
//         t_result = ttnn::add(nan_res, t_result, std::nullopt, output_mem_config);
//     }
//     // input == 1, output is 0
//     Tensor result = where(ttnn::eq(input_a, t_one, std::nullopt, output_mem_config), 0.0f, t_result, output_mem_config);
//     return result;
// }

// // asinh(x) = log(x + sqrt(x^2 + 1))
// Tensor _asinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor ln_res(input_a);
//     {
//         Tensor x_abs = abs(input_a, output_mem_config);
//         Tensor x_sq_p1(input_a);
//         {
//             Tensor x_sq = ttnn::square(input_a, output_mem_config);
//             x_sq_p1 = add_unary(x_sq, 1.0f, output_mem_config);
//         }
//         ln_res =
//             log(ttnn::add(x_abs, sqrt(x_sq_p1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
//     }
//     // input is negative, output is -asinh(input)
//     Tensor result = where(input_a, ln_res, neg(ln_res, output_mem_config), output_mem_config);
//     return result;
// }

// // atanh[x] = 0.5 * ln((1 + x) / (1 - x))
// Tensor _atanh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor comp_result(input_a);
//     {
//         Tensor nr_term(input_a);
//         {
//             Tensor pos_x = add_unary(input_a, 1.0f, output_mem_config);
//             Tensor neg_x = sub_unary(input_a, 1.0f, output_mem_config);
//             nr_term = log(
//                 ttnn::multiply(
//                     pos_x, recip(neg(neg_x, output_mem_config), output_mem_config), std::nullopt, output_mem_config),
//                 output_mem_config);
//         }
//         comp_result = mul_unary(nr_term, 0.5f, output_mem_config);
//     }
//     // Input is -1 > value > 1, output is nan
//     // Input is -1 < value < 1, output is atanh(input)
//     float t_nan = std::nanf("");
//     Tensor abs_temp = sub_unary(abs(input_a, output_mem_config), 1.0f, output_mem_config);
//     Tensor result = where(ltz(abs_temp, output_mem_config), comp_result, t_nan, output_mem_config);
//     return result;
// }

// // cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
// //         = exp[ (1/3)*log[a] ]
// Tensor _cbrt(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
//     constexpr float scale = (float)(1.0 / 3.0);
//     Tensor t_scale =
//         ttnn::operations::creation::create_scalar(scale, input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
//     Tensor t_ln_input =
//         log(abs(input_tensor, output_mem_config), output_mem_config);  // negative log is not useful here
//     Tensor t1 = ttnn::multiply(t_ln_input, t_scale, std::nullopt, output_mem_config);
//     t_scale.deallocate();
//     t_ln_input.deallocate();

// // acosh(x) = log(x + sqrt(x^2 - 1))
// Tensor _acosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor t_one = ones_like(input_a, output_mem_config);
//     Tensor t_result(input_a);
//     {
//         Tensor ln_res(input_a);
//         {
//             Tensor x_abs = abs(input_a, output_mem_config);
//             Tensor x_sq_m1(input_a);
//             {
//                 Tensor x_sq = ttnn::square(x_abs, output_mem_config);
//                 x_sq_m1 = sub_unary(x_sq, 1.0f, output_mem_config);
//             }
//             ln_res = log(
//                 ttnn::add(x_abs, sqrt(x_sq_m1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
//         }
//         // To handle inputs <= 1
//         // input < 1, output is nan
//         // input > 1, output is acosh(input)
//         Tensor scalar = ttnn::operations::creation::create_scalar(
//             std::nanf(""), input_a.get_dtype(), Layout::TILE, input_a.device());
//         Tensor nan_res = ttnn::multiply(
//             ttnn::le(input_a, t_one, std::nullopt, output_mem_config), scalar, std::nullopt, output_mem_config);
//         scalar.deallocate();
//         t_result = ttnn::multiply(
//             ttnn::gt(input_a, t_one, std::nullopt, output_mem_config), ln_res, std::nullopt, output_mem_config);
//         t_result = ttnn::add(nan_res, t_result, std::nullopt, output_mem_config);
//     }
//     // input == 1, output is 0
//     Tensor result = where(ttnn::eq(input_a, t_one, std::nullopt, output_mem_config), 0.0f, t_result, output_mem_config);
//     return result;
// }

// // asinh(x) = log(x + sqrt(x^2 + 1))
// Tensor _asinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor ln_res(input_a);
//     {
//         Tensor x_abs = abs(input_a, output_mem_config);
//         Tensor x_sq_p1(input_a);
//         {
//             Tensor x_sq = ttnn::square(input_a, output_mem_config);
//             x_sq_p1 = add_unary(x_sq, 1.0f, output_mem_config);
//         }
//         ln_res =
//             log(ttnn::add(x_abs, sqrt(x_sq_p1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
//     }
//     // input is negative, output is -asinh(input)
//     Tensor result = where(input_a, ln_res, neg(ln_res, output_mem_config), output_mem_config);
//     return result;
// }

// // atanh[x] = 0.5 * ln((1 + x) / (1 - x))
// Tensor _atanh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor comp_result(input_a);
//     {
//         Tensor nr_term(input_a);
//         {
//             Tensor pos_x = add_unary(input_a, 1.0f, output_mem_config);
//             Tensor neg_x = sub_unary(input_a, 1.0f, output_mem_config);
//             nr_term = log(
//                 ttnn::multiply(
//                     pos_x, recip(neg(neg_x, output_mem_config), output_mem_config), std::nullopt, output_mem_config),
//                 output_mem_config);
//         }
//         comp_result = mul_unary(nr_term, 0.5f, output_mem_config);
//     }
//     // Input is -1 > value > 1, output is nan
//     // Input is -1 < value < 1, output is atanh(input)
//     float t_nan = std::nanf("");
//     Tensor abs_temp = sub_unary(abs(input_a, output_mem_config), 1.0f, output_mem_config);
//     Tensor result = where(ltz(abs_temp, output_mem_config), comp_result, t_nan, output_mem_config);
//     return result;
// }

// // cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
// //         = exp[ (1/3)*log[a] ]
// Tensor _cbrt(const Tensor& input_tensor, const MemoryConfig& output_mem_config) {
//     constexpr float scale = (float)(1.0 / 3.0);
//     Tensor t_scale =
//         ttnn::operations::creation::create_scalar(scale, input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
//     Tensor t_ln_input =
//         log(abs(input_tensor, output_mem_config), output_mem_config);  // negative log is not useful here
//     Tensor t1 = ttnn::multiply(t_ln_input, t_scale, std::nullopt, output_mem_config);
//     t_scale.deallocate();
//     t_ln_input.deallocate();
//     Tensor t2 = exp(t1, output_mem_config);
//     t1.deallocate();
//     Tensor t3 = ttnn::multiply(t2, sign(input_tensor, output_mem_config), std::nullopt, output_mem_config);
//     return t3;
// }

// // cosh[x] = (exp[x] + exp[-x])/2
// Tensor _cosh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor e_pos_x = exp(input_a, output_mem_config);
//     Tensor e_neg_x = exp(neg(input_a, output_mem_config), output_mem_config);
//     Tensor nr_term = ttnn::add(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
//     e_pos_x.deallocate();
//     e_neg_x.deallocate();
//     Tensor scalar =
//         ttnn::operations::creation::create_scalar(0.5f, input_a.get_dtype(), Layout::TILE, input_a.device());
//     return ttnn::multiply(nr_term, scalar, std::nullopt, output_mem_config);
//     scalar.deallocate();
// }

// // TODO: In future will uplift the op once the floor and tan has supported.
// // digamma support for the range of (1, inf)
// Tensor _digamma(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor t_log_out = log(input_a, output_mem_config);  // negative log is not useful here

//     // 1/2(z)
//     Tensor output = mul_unary(recip(input_a, output_mem_config), 0.5f, output_mem_config);
//     Tensor tmp = ttnn::square(recip(input_a, output_mem_config), output_mem_config);
//     Tensor val_square = tmp;
//     // (1/12) * x^2
//     output = ttnn::subtract(output, mul_unary(tmp, 0.083333333f, output_mem_config), std::nullopt, output_mem_config);

//     // (1/120) * x^4
//     tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
//     output =
//         ttnn::add(output, mul_unary(tmp, 0.008333333333333333f, output_mem_config), std::nullopt, output_mem_config);

//     //(1/252) * x^6
//     tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
//     output = ttnn::subtract(
//         output, mul_unary(tmp, 0.003968253968253968f, output_mem_config), std::nullopt, output_mem_config);

//     // (1/240) *x^8
//     tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
//     output =
//         ttnn::add(output, mul_unary(tmp, 0.004166666666666667f, output_mem_config), std::nullopt, output_mem_config);

//     //(1/132) * x^10
//     tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
//     output = ttnn::subtract(
//         output, mul_unary(tmp, 0.007575757575757576, output_mem_config), std::nullopt, output_mem_config);

//     //(691/32760) * x^12
//     tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
//     output =
//         ttnn::add(output, mul_unary(tmp, 0.021092796092796094, output_mem_config), std::nullopt, output_mem_config);

//     //(1/12) * x^14
//     tmp = ttnn::multiply(tmp, val_square, std::nullopt, output_mem_config);
//     output =
//         ttnn::subtract(output, mul_unary(tmp, 0.08333333333333333, output_mem_config), std::nullopt, output_mem_config);

//     return ttnn::subtract(t_log_out, output, std::nullopt, output_mem_config);
// }

// // hardtanh
// Tensor _hardtanh(
//     const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
//     return clip(a, low, high, output_mem_config);
// }

// // Function Clip
// // use clip y = min( max( x, min_value), max_value) by broadcast
// // Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
// Tensor _clip(const Tensor& a, float low, float high, const MemoryConfig& output_mem_config) {
//     const Tensor h_const = full_like(a, high);
//     Tensor a_max = tt::tt_metal::min(a, h_const, output_mem_config);
//     if (low == 0.0f) {
//         return relu(a_max, output_mem_config);
//     } else {
//         const Tensor l_const = full_like(a, low);
//         return tt::tt_metal::max(a_max, l_const, output_mem_config);
//     }
// }
// Tensor clip(const Tensor& a, float low, float high, const MemoryConfig& output_mem_config) {
//     return operation::decorate_as_composite(__func__, _clip)(a, low, high, output_mem_config);
// }

// Tensor _lgamma(const Tensor& x, const MemoryConfig& output_mem_config) {
//     Tensor result(x);
//     {
//         Tensor t(x);
//         {
//             Tensor temp_log(x);
//             {
//                 Tensor temp(x);
//                 Tensor input = sub_unary(x, 1.0f, output_mem_config);
//                 {
//                     Tensor z1 = mul_unary(
//                         recip(add_unary(input, 1.0f, output_mem_config), output_mem_config),
//                         76.18009172947146f,
//                         output_mem_config);
//                     temp = add_unary(z1, 1.0f, output_mem_config);

//                     z1 = mul_unary(
//                         recip(add_unary(input, 2.0f, output_mem_config), output_mem_config),
//                         -86.50532032941677f,
//                         output_mem_config);
//                     temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

//                     z1 = mul_unary(
//                         recip(add_unary(input, 3.0f, output_mem_config), output_mem_config),
//                         24.01409824083091f,
//                         output_mem_config);
//                     temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

//                     z1 = mul_unary(
//                         recip(add_unary(input, 4.0f, output_mem_config), output_mem_config),
//                         -1.231739572450155f,
//                         output_mem_config);
//                     temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

//                     z1 = mul_unary(
//                         recip(add_unary(input, 5.0f, output_mem_config), output_mem_config),
//                         0.1208650973866179e-2f,
//                         output_mem_config);
//                     temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

//                     z1 = mul_unary(
//                         recip(add_unary(input, 6.0f, output_mem_config), output_mem_config),
//                         -0.5395239384953e-5f,
//                         output_mem_config);
//                     temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);
//                 }
//                 {
//                     Tensor t_log(x);
//                     {
//                         t = add_unary(input, 5.5f, output_mem_config);
//                         t_log = log(t, output_mem_config);
//                     }
//                     temp_log = log(temp, output_mem_config);
//                     result = add_unary(
//                         ttnn::multiply(
//                             add_unary(input, 0.5f, output_mem_config), t_log, std::nullopt, output_mem_config),
//                         0.918938531357171f,
//                         output_mem_config);
//                 }
//             }
//             result = ttnn::add(result, temp_log, std::nullopt, output_mem_config);
//         }
//         result = ttnn::subtract(result, t, std::nullopt, output_mem_config);
//         {
//             {
//                 Tensor t_one = ones_like(x, output_mem_config);
//                 result = where(ttnn::eq(x, t_one, std::nullopt, output_mem_config), 0.0f, result, output_mem_config);
//             }
//             {
//                 Tensor t_two = mk_filled_tensor_like(x, 2.0f, output_mem_config);
//                 result = where(ttnn::eq(x, t_two, std::nullopt, output_mem_config), 0.0f, result, output_mem_config);
//             }
//         }
//     }
//     return result;
// }


// // log1p 1
// // use transformation y = log(1.0 + x) by broadcast
// Tensor _log1p(const Tensor& x, const MemoryConfig& output_mem_config) {
//     Tensor x_1 = add1(x, output_mem_config);
//     Tensor result_log1p = log(x_1, output_mem_config);
//     return result_log1p;
// }


// // mish[x] = x*tanh[softplus[x]]
// // use transformation y = x*tanh[softplus[x]] by broadcast
// // Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
// Tensor _mish(const Tensor& x, const MemoryConfig& output_mem_config) {
//     std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({x}))};
//     operation::launch_op(
//         [output_mem_config](
//             const std::vector<Tensor>& input_tensors,
//             const std::vector<std::optional<const Tensor>>& optional_input_tensors,
//             const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
//             const auto& x = input_tensors.at(0);
//             Tensor sp_x = softplus(x, 1.0f, 20.0f, output_mem_config);
//             Tensor tanh_x = tanh(sp_x, output_mem_config);
//             sp_x.deallocate();
//             Tensor mish_x = ttnn::multiply(x, tanh_x, std::nullopt, output_mem_config);
//             return {mish_x};
//         },
//         {x},
//         output_tensors);
//     return output_tensors.at(0);
// }

// // multivariate log-gamma function
// // Ref : https://pytorch.org/docs/stable/special.html#torch.special.multigammaln
// Tensor _multigammaln(const Tensor& x, const MemoryConfig& output_mem_config) {
//     Tensor result = lgamma(x, output_mem_config);
//     result = ttnn::add(
//         result, lgamma(sub_unary(x, 0.5f, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
//     result = ttnn::add(
//         result, lgamma(sub_unary(x, 1.0f, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
//     result = ttnn::add(
//         result, lgamma(sub_unary(x, 1.5f, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
//     result = add_unary(result, 3.434189657547f, output_mem_config);
//     return result;
// }


// // sinh[x] = (exp[x] - exp[-x])/2
// Tensor _sinh(const Tensor& input_a, const MemoryConfig& output_mem_config) {
//     Tensor e_pos_x = exp(input_a, output_mem_config);
//     Tensor e_neg_x = exp(neg(input_a, output_mem_config), output_mem_config);
//     Tensor nr_term = ttnn::subtract(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
//     e_pos_x.deallocate();
//     e_neg_x.deallocate();
//     Tensor scalar =
//         ttnn::operations::creation::create_scalar(0.5f, input_a.get_dtype(), Layout::TILE, input_a.device());
//     return ttnn::multiply(nr_term, scalar, std::nullopt, output_mem_config);
//     scalar.deallocate();
// }


// // Function: softsign
// // Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
// Tensor _softsign(const Tensor& a, const MemoryConfig& output_mem_config) {
//     return ttnn::multiply(
//         a,
//         recip(add1(abs(a, output_mem_config), output_mem_config), output_mem_config),
//         std::nullopt,
//         output_mem_config);
// }


// Tensor _swish(const Tensor& a, const MemoryConfig& output_mem_config) {
//     // x / (1.0f + exp(-x))
//     return silu(a, output_mem_config);
// }

// tanhshrink(x) = x - tanh(x)
Tensor _tanhshrink(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    std::cout<<"\n\n hit in ttnn tanhshrink";
    Tensor tan_x = ttnn::tanh(x, output_mem_config);
    Tensor result = ttnn::subtract(x, tan_x, std::nullopt, output_mem_config);
    return result;
}

// // Function @hard_swish
// // use transformation y = x * hardsigmoid( x ) by broadcast
// // Ref: PyTorch
// // hard swish(x) = x*hardsigmoid(x,scale,shift)
// Tensor _hardswish(const Tensor& a, float scale, float shift, const MemoryConfig& output_mem_config) {
//     Tensor a_sigmoid = hardsigmoid(a, scale, shift, output_mem_config);
//     Tensor result_sq = ttnn::multiply(a_sigmoid, a, std::nullopt, output_mem_config);
//     return result_sq;
// }

// Tensor _hardsigmoid(const Tensor& a, float scale, float shift, const MemoryConfig& output_mem_config) {
//     Tensor a_mac = mac(a, scale, shift, output_mem_config);  // multiply and add.
//     Tensor a_clip = relu_max(a_mac, 1.0f, output_mem_config);
//     return a_clip;
// }

// Tensor _tril(const Tensor& input_a, int32_t diag, const MemoryConfig& output_mem_config) {
//     Tensor index_l = tt::numpy::index_tril<bfloat16>(
//         input_a.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
//     return ttnn::multiply(input_a, index_l, std::nullopt, output_mem_config);
// }


// // triu : select upper triangular region of input matrix
// Tensor _triu(const Tensor& input_a, int32_t diag, const MemoryConfig& output_mem_config) {
//     Tensor index_u = tt::numpy::index_triu<bfloat16>(
//         input_a.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input_a.device(), output_mem_config);
//     return ttnn::multiply(input_a, index_u, std::nullopt, output_mem_config);
// }

std::function<ttnn::Tensor(const Tensor&, const std::optional<MemoryConfig>&)> get_function_type1(UnaryCompositeOpType OpType){
    std::cout<<"\n\n inside get fun 1";
    switch (OpType) {
        // case UnaryCompositeOpType::ACOSH:
        //     return _acosh;
        // case UnaryCompositeOpType::ASINH:
        //     return _asinh;
        // case UnaryCompositeOpType::ATANH:
        //     return _atanh;
        // case UnaryCompositeOpType::CBRT:
        //     return _cbrt;
        // case UnaryCompositeOpType::COSH:
        //     return _cosh;
        // case UnaryCompositeOpType::DIGAMMA:
        //     return _digamma;
        // case UnaryCompositeOpType::HARDTANH:
        //     return _hardtanh;
        // case UnaryCompositeOpType::LGAMMA:
        //     return _lgamma;
        // case UnaryCompositeOpType::LOG1P:
        //     return _log1p;
        // case UnaryCompositeOpType::MISH:
        //     return _mish;
        // case UnaryCompositeOpType::MULTIGAMMALN:
        //     return _multigammaln;
        // case UnaryCompositeOpType::SINH:
        //     return _sinh;
        // case UnaryCompositeOpType::SOFTSIGN:
        //     return _softsign;
        // case UnaryCompositeOpType::SWISH:
        //     return _swish;
        case UnaryCompositeOpType::TANHSHRINK:
            return _tanhshrink;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}


std::function<ttnn::Tensor(const Tensor&, float, float, const std::optional<MemoryConfig>&)> get_function_type2(UnaryCompositeOpType OpType){
    switch (OpType) {
        // case UnaryCompositeOpType::HARDSWISH:
        //     return _hardswish;
        // case UnaryCompositeOpType::HARDSIGMOID:
        //     return _hardsigmoid;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<ttnn::Tensor(const Tensor&, int, const std::optional<MemoryConfig>&)> get_function_type3(UnaryCompositeOpType OpType){
    switch (OpType) {
        // case UnaryCompositeOpType::TRIL:
        //     return _tril;
        // case UnaryCompositeOpType::TRIU:
        //     return _triu;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

}
}  // namespace ttnn::operations::unary

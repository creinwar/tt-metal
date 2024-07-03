// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/complex/complex_ops.hpp"
#include "tt_dnn/op_library/concat/concat_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace tt {

namespace tt_metal {

//TODO: add profiling hooks

#define CHECK_FOR_COMPLEX(input) do {\
  TT_ASSERT( utility::is_complex_shape(input), "works for complex shape only"); \
  /* TT_ASSERT( input.get_legacy_shape()[0] == 1, "tensor should have batch size 1"); */ \
  } while(0);

Tensor mk_complex(const Tensor& input_r, const Tensor& input_i, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> inputs = {input_r,input_i};
    return concat(inputs, -1, output_mem_config);
}

Tensor get_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    Shape t_Shape = input.get_legacy_shape();
    Shape start = {0, 0, 0, 0} ;
    Shape end = {t_Shape[0] - 1,t_Shape[1] - 1 ,t_Shape[2] - 1, (t_Shape[3] / 2) - 1};
    Tensor r_tensor = unpad(input, start, end, output_mem_config);
    return r_tensor;
}

Tensor get_imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    Shape t_Shape = input.get_legacy_shape();
    Shape start = {0, 0, 0, (t_Shape[3] / 2)};
    Shape end = {t_Shape[0] - 1,t_Shape[1] - 1 ,t_Shape[2] - 1, (t_Shape[3] - 1)};
    Tensor i_tensor = unpad(input, start, end, output_mem_config);
    return i_tensor;
}

namespace utility {
    bool is_complex_shape(const Tensor& input) {
        const Shape& shape = input.get_legacy_shape();
        return shape[-1]%(2*TILE_WIDTH) == 0; //last dim should be partitionable
    }
}


Tensor _is_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    return eqz(real, output_mem_config); //imaginary portion = 0
}
Tensor is_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _is_real)(input, output_mem_config);
}

Tensor is_imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor imag = get_imag(input, output_mem_config);
    return eqz(imag, output_mem_config);
}

Tensor real(const Tensor& input, const MemoryConfig& output_mem_config) {
	CHECK_FOR_COMPLEX(input);
    return get_real(input, output_mem_config);
}

Tensor imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    return get_imag(input, output_mem_config);
}

Tensor conj(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);
    return mk_complex(real,neg(imag, output_mem_config));
}

Tensor complex_abs(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);
    return hypot(real, imag, output_mem_config);
}

Tensor complex_recip(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);

    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);

    Tensor a_plus_b = ttnn::add(real,imag, std::nullopt, output_mem_config);
    Tensor a_minus_b = ttnn::subtract(real, imag, std::nullopt,output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(ttnn::square(real,output_mem_config),ttnn::square(imag,output_mem_config), std::nullopt, output_mem_config);
    Tensor inv_dr = recip( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = ttnn::multiply( neg(imag,output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply( real, inv_dr, std::nullopt, output_mem_config);
    return mk_complex( conj_re, conj_im, output_mem_config );
}

Tensor complex_add(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    return ttnn::add(input_a,input_b, std::nullopt, output_mem_config);
}

Tensor complex_sub(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    return ttnn::subtract(input_a,input_b, std::nullopt, output_mem_config);
}

Tensor complex_mul(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input_a);
    CHECK_FOR_COMPLEX(input_b);

    Tensor r_a = get_real(input_a, output_mem_config);
    Tensor i_a = get_imag(input_a, output_mem_config);

    Tensor r_b = get_real(input_b, output_mem_config);
    Tensor i_b = get_imag(input_b, output_mem_config);

    Tensor re_part = ttnn::subtract(
        ttnn::multiply(r_a, r_b, std::nullopt, output_mem_config),
        ttnn::multiply(i_a, i_b, std::nullopt, output_mem_config),
        std::nullopt, output_mem_config);

    Tensor im_part = ttnn::add(
        ttnn::multiply(r_a,i_b,std::nullopt,output_mem_config),
        ttnn::multiply(i_a,r_b,std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);
    return mk_complex( re_part, im_part, output_mem_config);
}


// z_a/z_b = z_a*recip(z_b) = z_a*conj(z_b)/(z_b*conj(z_b))
Tensor complex_div(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input_a);
    CHECK_FOR_COMPLEX(input_b);
    return complex_mul( input_a, complex_recip( input_b , output_mem_config ), output_mem_config  );
}

// theta = /_x + iy = atan2(y,x)
Tensor angle(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);
    return neg( atan2(imag, real, output_mem_config), output_mem_config );
}

#undef CHECK_FOR_COMPLEX

///// type-2 implementation ////
Tensor is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return eqz( input[1], output_mem_config);
}

Tensor is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return eqz( input[0], output_mem_config);
}

Tensor real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[0];
}

Tensor imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[1];
}

ComplexTensor conj(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ComplexTensor({input[0], neg(input[1],output_mem_config)});
}

Tensor angle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return neg( atan2(input[1],input[0],output_mem_config), output_mem_config );
}

Tensor complex_abs(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return hypot(input[0],input[1],output_mem_config);
}

ComplexTensor complex_mul(const ComplexTensor& ab, const ComplexTensor& cd,  const MemoryConfig& output_mem_config) {
    // (a + ib)*(c + id) = (ac - bd) + i(bc + ad)
    Tensor re_part = ttnn::subtract(
        ttnn::multiply(ab[0],cd[0],std::nullopt,output_mem_config),
        ttnn::multiply(ab[1],cd[1],std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);

    Tensor im_part = ttnn::add(
        ttnn::multiply(ab[0],cd[1],std::nullopt,output_mem_config),
        ttnn::multiply(ab[1],cd[0],std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);

    return ComplexTensor({ re_part, im_part });
}

ComplexTensor complex_div(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config) {
    return complex_mul( input_a, complex_recip( input_b , output_mem_config ), output_mem_config  );
}

ComplexTensor complex_recip(const ComplexTensor& ab, const MemoryConfig& output_mem_config) {
    Tensor a_plus_b = ttnn::add(ab[0],ab[1],std::nullopt,output_mem_config);
    Tensor a_minus_b = ttnn::subtract(ab[0],ab[1],std::nullopt,output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(ttnn::square(ab[0],output_mem_config),ttnn::square(ab[1],output_mem_config),
                                std::nullopt,output_mem_config);
    Tensor inv_dr = recip( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = ttnn::multiply( neg(ab[1],output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply( ab[0], inv_dr, std::nullopt, output_mem_config);
    return ComplexTensor({ conj_re, conj_im});
}

ComplexTensor complex_add(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::add(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::add(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

ComplexTensor complex_sub(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::subtract(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::subtract(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

// level-1 type polar
Tensor polar(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor c = cos(input_b,output_mem_config);
    Tensor r = ttnn::multiply(input_a, c ,std::nullopt, output_mem_config);
    c.deallocate();

    Tensor s = sin(input_b,output_mem_config);
    Tensor i = ttnn::multiply(input_a, s, std::nullopt, output_mem_config);
    s.deallocate();
    return mk_complex( r, i, output_mem_config);
}

ComplexTensor polar(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    const Tensor& input_a = input.real();
    const Tensor& input_b = input.imag();
    Tensor c = cos(input_b,output_mem_config);
    Tensor r = ttnn::multiply(input_a,c,std::nullopt,output_mem_config);
    c.deallocate();

    Tensor s = sin(input_b,output_mem_config);
    Tensor i = ttnn::multiply(input_a,s,std::nullopt,output_mem_config);
    s.deallocate();

    return ComplexTensor({r,i});
}

// backward ops for type2 complex tensor
// complex conj
// self: grad.conj()
std::vector<ComplexTensor> conj_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_result = conj(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex imag
// imag: at::imag(grad)
std::vector<ComplexTensor> imag_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor r = zeros_like(real(input, output_mem_config), output_mem_config);
    ComplexTensor grad_result = ComplexTensor({r,grad});
    r.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex real
// real: at::real(grad)
std::vector<ComplexTensor> real_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor i = zeros_like(real(input, output_mem_config), output_mem_config);
    ComplexTensor grad_result = ComplexTensor({grad, i});
    i.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// complex sub
// self: grad, other: -grad * alpha
std::vector<ComplexTensor> complex_sub_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = grad;
    grad_tensor.emplace_back(grad);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    UnaryWithParam op1 {UnaryOpType::NEG};
    UnaryWithParam op2 {UnaryOpType::MUL_UNARY_SFPU, alpha};
    ComplexTensor grad_b = ComplexTensor({unary_chain( grad_r, {op1, op2}, output_mem_config), unary_chain( grad_i, {op1, op2}, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

// complex mul
// grad_input = grad * other.conj()
// grad_other = grad * input.conj()
std::vector<ComplexTensor> complex_mul_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = complex_mul(grad, conj(other,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    ComplexTensor grad_b = complex_mul(grad, conj(input,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

//  complex div
//  self: grad / other.conj();
//  other: -grad * ((self / other) / other).conj();
std::vector<ComplexTensor> complex_div_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(eqz(other.real(),output_mem_config), eqz(other.imag(),output_mem_config), std::nullopt, output_mem_config);
    ComplexTensor grad_a = complex_div(grad, conj(other,output_mem_config), output_mem_config);
    Tensor grad_a_r = where(condition_nan, full_like(grad.real(), std::nanf(""), output_mem_config), real(grad_a,output_mem_config),  output_mem_config);
    Tensor grad_a_i = where(condition_nan, full_like(grad.imag(), std::nanf(""), output_mem_config), imag(grad_a,output_mem_config),  output_mem_config);
    grad_a = ComplexTensor({grad_a_r, grad_a_i});
    grad_a_r.deallocate();
    grad_a_i.deallocate();
    grad_tensor.emplace_back(grad_a);
    ComplexTensor neg_grad = ComplexTensor({neg(grad.real(),output_mem_config), neg(grad.imag(),output_mem_config)});
    ComplexTensor grad_b = complex_mul(neg_grad, conj(complex_div(complex_div(input, other, output_mem_config), other, output_mem_config ),output_mem_config), output_mem_config);
    neg_grad.deallocate();
    Tensor grad_b_r = where(condition_nan, full_like(grad.real(), std::nanf(""), output_mem_config), real(grad_b,output_mem_config),  output_mem_config);
    Tensor grad_b_i = where(condition_nan, full_like(grad.imag(), std::nanf(""), output_mem_config), imag(grad_b,output_mem_config),  output_mem_config);
    grad_b = ComplexTensor({grad_b_r, grad_b_i});
    grad_b_r.deallocate();
    grad_b_i.deallocate();
    condition_nan.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

// complex abs
// self: grad * self.sgn()
std::vector<ComplexTensor> complex_abs_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor result = complex_abs(input, output_mem_config);
    Tensor grad_inp_r = where(eqz(result, output_mem_config), zeros_like(result, output_mem_config), ttnn::multiply(grad, ttnn::multiply(input.real(), recip(result, output_mem_config), std::nullopt, output_mem_config),std::nullopt, output_mem_config), output_mem_config );
    Tensor grad_inp_i = where(eqz(result, output_mem_config), zeros_like(result, output_mem_config), ttnn::multiply(grad, ttnn::multiply(input.imag(), recip(result, output_mem_config), std::nullopt, output_mem_config),std::nullopt, output_mem_config), output_mem_config );
    ComplexTensor grad_inp = ComplexTensor({ grad_inp_r, grad_inp_i});
    result.deallocate();
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}

// complex reciprocal
// self: -grad * (result * result).conj()
std::vector<ComplexTensor> complex_recip_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(eqz(input.real(),output_mem_config), eqz(input.imag(),output_mem_config), std::nullopt, output_mem_config);
    ComplexTensor neg_grad = ComplexTensor({neg(grad.real(),output_mem_config), neg(grad.imag(),output_mem_config)});
    ComplexTensor inp_recip = complex_recip(input, output_mem_config);
    ComplexTensor grad_inp = complex_mul(neg_grad, conj(complex_mul(inp_recip, inp_recip, output_mem_config), output_mem_config), output_mem_config) ;
    neg_grad.deallocate();
    inp_recip.deallocate();
    Tensor grad_inp_r = where(condition_nan, full_like(input.real(), std::nanf(""), output_mem_config), grad_inp.real(), output_mem_config);
    Tensor grad_inp_i = where(condition_nan, full_like(input.imag(), std::nanf(""), output_mem_config), grad_inp.imag(), output_mem_config);
    condition_nan.deallocate();
    grad_inp = ComplexTensor({ grad_inp_r, grad_inp_i});
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}

// angle at::where(self == 0.0, at::zeros({}, self.options()), grad * self / self.abs().pow(2)
std::vector<ComplexTensor> angle_bw(const Tensor& grad, const ComplexTensor& input, bool is_complextensor, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    const Tensor &inp_r = input.real();
    const Tensor &inp_i = input.imag();
    Tensor condition_zero = ttnn::logical_and(eqz(input.real(),output_mem_config), eqz(input.imag(),output_mem_config), std::nullopt, output_mem_config);
    Tensor abs_squared = recip(ttnn::add(ttnn::square(inp_r, output_mem_config), ttnn::square(inp_i, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    Tensor res_real = where(condition_zero, zeros_like(inp_r, output_mem_config), ttnn::multiply(grad, ttnn::multiply(neg(inp_i, output_mem_config), abs_squared, std::nullopt, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    Tensor res_imag = where(condition_zero, zeros_like(inp_i, output_mem_config), ttnn::multiply(grad, ttnn::multiply(inp_r, abs_squared, std::nullopt, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
    condition_zero.deallocate();
    abs_squared.deallocate();
    ComplexTensor grad_result = ComplexTensor({res_real, res_imag});
    res_real.deallocate();
    res_imag.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}


// polar
// grad_abs = torch.real(grad_conj * torch.sgn(result))
// result_mul_1_j = result * torch.tensor(0.0 + 1.0j)
// grad_angle = torch.real(grad_conj * result_mul_1_j)
// polar fwd op uses sin and cos hence input_b range is (0, 2*pi)
std::vector<ComplexTensor> polar_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor result = polar(input, output_mem_config);
    Tensor abs_result = complex_abs(result, output_mem_config);
    Tensor sgn_result_r = where(eqz(abs_result, output_mem_config), zeros_like(result.real(), output_mem_config), ttnn::multiply(result.real(), recip(abs_result, output_mem_config), std::nullopt, output_mem_config), output_mem_config );
    Tensor sgn_result_i = where(eqz(abs_result, output_mem_config), zeros_like(result.imag(), output_mem_config), ttnn::multiply(result.imag(), recip(abs_result, output_mem_config), std::nullopt, output_mem_config), output_mem_config );
    abs_result.deallocate();
    ComplexTensor sgn_result = ComplexTensor({ sgn_result_r, sgn_result_i });
    sgn_result_r.deallocate();
    sgn_result_i.deallocate();
    Tensor grad_abs = real(complex_mul(conj(grad, output_mem_config), sgn_result, output_mem_config), output_mem_config);
    sgn_result.deallocate();
    ComplexTensor flip_tensor = ComplexTensor({zeros_like(input.real(), output_mem_config), full_like(input.imag(), 1.0, output_mem_config) });
    Tensor grad_angle = real(complex_mul(conj(grad, output_mem_config), complex_mul(result, flip_tensor, output_mem_config), output_mem_config), output_mem_config);
    result.deallocate();
    flip_tensor.deallocate();
    ComplexTensor grad_result = ComplexTensor({grad_abs, grad_angle});
    grad_abs.deallocate();
    grad_angle.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

}//namespace tt_metal

}//namespace tt

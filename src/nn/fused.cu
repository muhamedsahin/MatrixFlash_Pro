#include "matrix_pro/nn/fused.hpp"
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/detail/fused_ops.cuh"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"

namespace matrix_pro {

template <typename Op>
Matrix fused_binary(const Matrix& a, const Matrix& b, Op op) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw ShapeMismatchError("fused_binary requires matching shapes");
    Matrix out(a.rows(), a.cols());
    fused_binary_(out, a, b, op);
    out.mark_host_stale();
    return out;
}

template <typename Op>
Matrix& fused_binary_(Matrix& out, const Matrix& a, const Matrix& b, Op op) {
    if (a.rows() != b.rows() || a.cols() != b.cols() ||
        out.rows() != a.rows() || out.cols() != a.cols())
        throw ShapeMismatchError("fused_binary_ requires matching shapes");
    detail::launch_binary(a.device_data(), b.device_data(), out.device_data(),
                          a.size(), op, compute_stream());
    checkCuda(cudaGetLastError(), "fused binary kernel launch");
    out.mark_host_stale();
    return out;
}

template <typename Op>
Matrix fused_ternary(const Matrix& a, const Matrix& b, const Matrix& c, Op op) {
    if (a.rows() != b.rows() || a.cols() != b.cols() ||
        a.rows() != c.rows() || a.cols() != c.cols())
        throw ShapeMismatchError("fused_ternary requires matching shapes");
    Matrix out(a.rows(), a.cols());
    detail::launch_ternary(a.device_data(), b.device_data(), c.device_data(),
                           out.device_data(), a.size(), op, compute_stream());
    checkCuda(cudaGetLastError(), "fused ternary kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix fused_sigmoid_mul(const Matrix& x, const Matrix& y) {
    return fused_binary(x, y, detail::SigmoidMulOp{});
}

Matrix fused_relu_add(const Matrix& x, const Matrix& y) {
    return fused_binary(x, y, detail::ReluAddOp{});
}

Matrix fused_add_mul(const Matrix& x, const Matrix& y, const Matrix& z) {
    return fused_ternary(x, y, z, detail::AddMulOp{});
}


Matrix fused_scale_bias(const Matrix& x, float scale, float bias) {
    Matrix out(x.rows(), x.cols());
    detail::launch_unary(x.device_data(), out.device_data(), x.size(),
                         detail::ScaleBiasOp{scale, bias}, compute_stream());
    checkCuda(cudaGetLastError(), "fused scale+bias launch");
    out.mark_host_stale();
    return out;
}

Matrix fused_bias_gelu(const Matrix& x, const Matrix& bias) {
    return fused_binary(x, bias, detail::BiasGeluOp{});
}

} // namespace matrix_pro

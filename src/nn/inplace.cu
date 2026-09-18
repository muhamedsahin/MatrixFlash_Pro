#include "matrix_pro/nn/inplace.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"
#include "matrix_pro/ops/operations.hpp"

#include <cuda_runtime.h>

namespace matrix_pro {
namespace {

struct AddOp { __device__ float operator()(float a, float b) const { return a + b; } };
struct SubOp { __device__ float operator()(float a, float b) const { return a - b; } };
struct MulOp { __device__ float operator()(float a, float b) const { return a * b; } };
struct DivOp { __device__ float operator()(float a, float b) const { return a / b; } };
struct AddScalarOp { float v; __device__ float operator()(float a) const { return a + v; } };
struct MulScalarOp { float v; __device__ float operator()(float a) const { return a * v; } };
struct NegOp { __device__ float operator()(float a) const { return -a; } };
struct ClampOp { float lo; float hi; __device__ float operator()(float a) const { return fminf(fmaxf(a, lo), hi); } };
struct ReluOp { __device__ float operator()(float a) const { return a > 0.0f ? a : 0.0f; } };
struct SigmoidOp { __device__ float operator()(float a) const { return 1.0f / (1.0f + expf(-a)); } };
struct TanhOp { __device__ float operator()(float a) const { return tanhf(a); } };
struct LeakyOp { float s; __device__ float operator()(float a) const { return a > 0.0f ? a : s * a; } };
struct EluOp { float a_; __device__ float operator()(float x) const { return x > 0.0f ? x : a_ * (expf(x) - 1.0f); } };
struct GeluOp {
    __device__ float operator()(float v) const {
        const float x3 = v * v * v;
        return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * x3)));
    }
};
struct SwishOp { float b; __device__ float operator()(float x) const { return x / (1.0f + expf(-b * x)); } };

__global__ void broadcast_inplace_kernel(float* self, const float* rhs,
                                         std::size_t rows, std::size_t cols,
                                         std::size_t rhs_rows, std::size_t rhs_cols, int is_add) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) return;
    const std::size_t r = index / cols;
    const std::size_t c = index % cols;
    const float y = rhs[(r % rhs_rows) * rhs_cols + (c % rhs_cols)];
    self[index] = is_add ? (self[index] + y) : (self[index] * y);
}

void require_same_shape(const Matrix& a, const Matrix& b, const char* what) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw ShapeMismatchError(what);
}

void check_broadcast_shapes(const Matrix& self, const Matrix& rhs) {
    const bool rows_ok = (rhs.rows() == self.rows() || rhs.rows() == 1);
    const bool cols_ok = (rhs.cols() == self.cols() || rhs.cols() == 1);
    if (!rows_ok || !cols_ok || rhs.size() == 0)
        throw ShapeMismatchError("broadcast in-place rhs must be 1x1, 1xC, Rx1 or RxC");
}

} // namespace

Matrix& add_(Matrix& self, const Matrix& other) {
    require_same_shape(self, other, "add_ shape mismatch");
    detail::launch_binary(self.device_data(), other.device_data(), self.device_data(),
                          self.size(), AddOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "add_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& subtract_(Matrix& self, const Matrix& other) {
    require_same_shape(self, other, "subtract_ shape mismatch");
    detail::launch_binary(self.device_data(), other.device_data(), self.device_data(),
                          self.size(), SubOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "subtract_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& elementwise_multiply_(Matrix& self, const Matrix& other) {
    require_same_shape(self, other, "elementwise_multiply_ shape mismatch");
    detail::launch_binary(self.device_data(), other.device_data(), self.device_data(),
                          self.size(), MulOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "elementwise_multiply_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& divide_(Matrix& self, const Matrix& other) {
    require_same_shape(self, other, "divide_ shape mismatch");
    detail::launch_binary(self.device_data(), other.device_data(), self.device_data(),
                          self.size(), DivOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "divide_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& add_scalar_(Matrix& self, float value) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         AddScalarOp{value}, compute_stream());
    checkCuda(cudaGetLastError(), "add_scalar_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& multiply_scalar_(Matrix& self, float value) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         MulScalarOp{value}, compute_stream());
    checkCuda(cudaGetLastError(), "multiply_scalar_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& negate_(Matrix& self) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         NegOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "negate_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& clamp_(Matrix& self, float low, float high) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         ClampOp{low, high}, compute_stream());
    checkCuda(cudaGetLastError(), "clamp_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& relu_(Matrix& self) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         ReluOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "relu_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& sigmoid_(Matrix& self) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         SigmoidOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "sigmoid_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& tanh_(Matrix& self) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         TanhOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "tanh_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& leaky_relu_(Matrix& self, float negative_slope) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         LeakyOp{negative_slope}, compute_stream());
    checkCuda(cudaGetLastError(), "leaky_relu_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& elu_(Matrix& self, float alpha) {
    if (alpha < 0.0f) throw InvalidArgumentError("ELU alpha must be non-negative");
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         EluOp{alpha}, compute_stream());
    checkCuda(cudaGetLastError(), "elu_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& gelu_(Matrix& self) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         GeluOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "gelu_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& swish_(Matrix& self, float beta) {
    detail::launch_unary(self.device_data(), self.device_data(), self.size(),
                         SwishOp{beta}, compute_stream());
    checkCuda(cudaGetLastError(), "swish_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& broadcast_add_(Matrix& self, const Matrix& rhs) {
    check_broadcast_shapes(self, rhs);
    broadcast_inplace_kernel<<<detail::grid_blocks(self.size()), 256, 0, compute_stream()>>>(
        self.device_data(), rhs.device_data(),
        self.rows(), self.cols(), rhs.rows(), rhs.cols(), 1);
    checkCuda(cudaGetLastError(), "broadcast_add_ kernel launch");
    self.mark_host_stale();
    return self;
}
Matrix& broadcast_multiply_(Matrix& self, const Matrix& rhs) {
    check_broadcast_shapes(self, rhs);
    broadcast_inplace_kernel<<<detail::grid_blocks(self.size()), 256, 0, compute_stream()>>>(
        self.device_data(), rhs.device_data(),
        self.rows(), self.cols(), rhs.rows(), rhs.cols(), 0);
    checkCuda(cudaGetLastError(), "broadcast_multiply_ kernel launch");
    self.mark_host_stale();
    return self;
}

} // namespace matrix_pro


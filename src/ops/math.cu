#include <cuda_runtime.h>
#include "matrix_pro/ops/math.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/detail/kernel_helpers.cuh"
#include "matrix_pro/detail/fused_ops.cuh"

namespace matrix_pro {

// Unary Operators
struct SinOp { __device__ float operator()(float x) const { return __sinf(x); } };
struct CosOp { __device__ float operator()(float x) const { return __cosf(x); } };
struct TanOp { __device__ float operator()(float x) const { return __tanf(x); } };
struct AsinOp { __device__ float operator()(float x) const { return asinf(x); } };
struct AcosOp { __device__ float operator()(float x) const { return acosf(x); } };
struct AtanOp { __device__ float operator()(float x) const { return atanf(x); } };
struct SinhOp { __device__ float operator()(float x) const { return sinhf(x); } };
struct CoshOp { __device__ float operator()(float x) const { return coshf(x); } };

struct CeilOp { __device__ float operator()(float x) const { return ceilf(x); } };
struct FloorOp { __device__ float operator()(float x) const { return floorf(x); } };
struct RoundOp { __device__ float operator()(float x) const { return roundf(x); } };
struct TruncOp { __device__ float operator()(float x) const { return truncf(x); } };

struct ErfOp { __device__ float operator()(float x) const { return erff(x); } };
struct ErfcOp { __device__ float operator()(float x) const { return erfcf(x); } };

__device__ float erfinv_approx(float x) {
    float w = -logf((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w -= 2.5f;
        p = 2.81022636e-08f;
        p = fmaf(p, w, 3.43273939e-07f);
        p = fmaf(p, w, -3.5233877e-06f);
        p = fmaf(p, w, -4.39150654e-06f);
        p = fmaf(p, w, 0.00021858087f);
        p = fmaf(p, w, -0.00125372503f);
        p = fmaf(p, w, -0.00417768164f);
        p = fmaf(p, w, 0.246640727f);
        p = fmaf(p, w, 1.50140941f);
    } else {
        w = sqrtf(w) - 3.0f;
        p = -0.000200214257f;
        p = fmaf(p, w, 0.000100950558f);
        p = fmaf(p, w, 0.00134934322f);
        p = fmaf(p, w, -0.00367342844f);
        p = fmaf(p, w, 0.00573950773f);
        p = fmaf(p, w, -0.0076224613f);
        p = fmaf(p, w, 0.00943887047f);
        p = fmaf(p, w, 1.00167406f);
        p = fmaf(p, w, 2.83297682f);
    }
    return p * x;
}

struct ErfinvOp { __device__ float operator()(float x) const { return erfinv_approx(x); } };
struct LgammaOp { __device__ float operator()(float x) const { return lgammaf(x); } };

struct RcpOp { __device__ float operator()(float x) const { return __frcp_rn(x); } };
struct RsqrtOp { __device__ float operator()(float x) const { return __frsqrt_rn(x); } };

struct SignOp { 
    __device__ float operator()(float x) const { 
        if (x > 0.0f) return 1.0f;
        if (x < 0.0f) return -1.0f;
        return 0.0f;
    } 
};

// Binary Operators
struct Atan2Op { __device__ float operator()(float y, float x) const { return atan2f(y, x); } };
struct FmodOp { __device__ float operator()(float a, float b) const { return fmodf(a, b); } };
struct RemainderOp { __device__ float operator()(float a, float b) const { return remainderf(a, b); } };
struct MinimumOp { __device__ float operator()(float a, float b) const { return fminf(a, b); } };
struct MaximumOp { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } };

struct LerpScalarOp {
    float t;
    __device__ float operator()(float a, float b) const {
        return a + t * (b - a);
    }
};

// Ternary Operators & Kernels
template<typename F>
__global__ void ternary_float4_kernel(const float* a, const float* b, const float* c, float* out, size_t size, F op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t vec_size = size / 4;
    
    for (size_t i = idx; i < vec_size; i += stride) {
        float4 a4 = reinterpret_cast<const float4*>(a)[i];
        float4 b4 = reinterpret_cast<const float4*>(b)[i];
        float4 c4 = reinterpret_cast<const float4*>(c)[i];
        float4 out4;
        out4.x = op(a4.x, b4.x, c4.x);
        out4.y = op(a4.y, b4.y, c4.y);
        out4.z = op(a4.z, b4.z, c4.z);
        out4.w = op(a4.w, b4.w, c4.w);
        reinterpret_cast<float4*>(out)[i] = out4;
    }
    
    for (size_t i = vec_size * 4 + idx; i < size; i += stride) {
        out[i] = op(a[i], b[i], c[i]);
    }
}

template <typename F>
void launch_ternary(const float* a, const float* b, const float* c, float* out, size_t size, F op) {
    int threads = 256;
    int blocks = detail::capped_grid(size, threads);
    ternary_float4_kernel<<<blocks, threads, 0, compute_stream()>>>(a, b, c, out, size, op);
    checkCuda(cudaGetLastError(), "ternary_float4_kernel");
}

struct LerpOp {
    __device__ float operator()(float a, float b, float t) const {
        return a + t * (b - a);
    }
};

struct AddcmulOp {
    float alpha;
    __device__ float operator()(float self, float t1, float t2) const {
        return self + alpha * t1 * t2;
    }
};

struct AddcdivOp {
    float alpha;
    __device__ float operator()(float self, float t1, float t2) const {
        return self + alpha * (t1 / t2);
    }
};

// --- Implementations ---

#define IMPLEMENT_UNARY(func_name, op_struct) \
Matrix func_name(const Matrix& m) { \
    Matrix result(m.rows(), m.cols(), MemoryMode::device_only); \
    detail::launch_unary(m.device_data(), result.device_data(), m.elements(), op_struct{}, compute_stream()); \
    result.mark_host_stale(); \
    return result; \
}

#define IMPLEMENT_BINARY(func_name, op_struct) \
Matrix func_name(const Matrix& a, const Matrix& b) { \
    if (a.rows() != b.rows() || a.cols() != b.cols()) throw ShapeMismatchError(#func_name " requires matching shapes"); \
    Matrix result(a.rows(), a.cols(), MemoryMode::device_only); \
    detail::launch_binary(a.device_data(), b.device_data(), result.device_data(), a.elements(), op_struct{}, compute_stream()); \
    result.mark_host_stale(); \
    return result; \
}

// Unary
IMPLEMENT_UNARY(sin, SinOp)
IMPLEMENT_UNARY(cos, CosOp)
IMPLEMENT_UNARY(tan, TanOp)
IMPLEMENT_UNARY(asin, AsinOp)
IMPLEMENT_UNARY(acos, AcosOp)
IMPLEMENT_UNARY(atan, AtanOp)
IMPLEMENT_UNARY(sinh, SinhOp)
IMPLEMENT_UNARY(cosh, CoshOp)
IMPLEMENT_UNARY(ceil, CeilOp)
IMPLEMENT_UNARY(floor, FloorOp)
IMPLEMENT_UNARY(round, RoundOp)
IMPLEMENT_UNARY(trunc, TruncOp)
IMPLEMENT_UNARY(erf, ErfOp)
IMPLEMENT_UNARY(erfc, ErfcOp)
IMPLEMENT_UNARY(erfinv, ErfinvOp)
IMPLEMENT_UNARY(lgamma, LgammaOp)
IMPLEMENT_UNARY(reciprocal, RcpOp)
IMPLEMENT_UNARY(rsqrt, RsqrtOp)
IMPLEMENT_UNARY(sign, SignOp)

// Binary
IMPLEMENT_BINARY(atan2, Atan2Op)
IMPLEMENT_BINARY(fmod, FmodOp)
IMPLEMENT_BINARY(remainder, RemainderOp)
IMPLEMENT_BINARY(minimum, MinimumOp)
IMPLEMENT_BINARY(maximum, MaximumOp)

// Ternary & Custom Implementations

Matrix lerp(const Matrix& a, const Matrix& b, float t) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw ShapeMismatchError("lerp requires matching shapes");
    }
    Matrix result(a.rows(), a.cols(), MemoryMode::device_only);
    detail::launch_binary(a.device_data(), b.device_data(), result.device_data(), a.elements(), LerpScalarOp{t}, compute_stream());
    result.mark_host_stale();
    return result;
}

Matrix lerp(const Matrix& a, const Matrix& b, const Matrix& t) {
    if (a.rows() != b.rows() || a.cols() != b.cols() || a.rows() != t.rows() || a.cols() != t.cols()) {
        throw ShapeMismatchError("lerp requires matching shapes");
    }
    Matrix result(a.rows(), a.cols(), MemoryMode::device_only);
    launch_ternary(a.device_data(), b.device_data(), t.device_data(), result.device_data(), a.elements(), LerpOp{});
    result.mark_host_stale();
    return result;
}

Matrix addcmul(const Matrix& self, const Matrix& t1, const Matrix& t2, float alpha) {
    if (self.rows() != t1.rows() || self.cols() != t1.cols() || self.rows() != t2.rows() || self.cols() != t2.cols()) {
        throw ShapeMismatchError("addcmul requires matching shapes");
    }
    Matrix result(self.rows(), self.cols(), MemoryMode::device_only);
    launch_ternary(self.device_data(), t1.device_data(), t2.device_data(), result.device_data(), self.elements(), AddcmulOp{alpha});
    result.mark_host_stale();
    return result;
}

Matrix addcdiv(const Matrix& self, const Matrix& t1, const Matrix& t2, float alpha) {
    if (self.rows() != t1.rows() || self.cols() != t1.cols() || self.rows() != t2.rows() || self.cols() != t2.cols()) {
        throw ShapeMismatchError("addcdiv requires matching shapes");
    }
    Matrix result(self.rows(), self.cols(), MemoryMode::device_only);
    launch_ternary(self.device_data(), t1.device_data(), t2.device_data(), result.device_data(), self.elements(), AddcdivOp{alpha});
    result.mark_host_stale();
    return result;
}

} // namespace matrix_pro

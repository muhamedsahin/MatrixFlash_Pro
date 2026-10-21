/**
 * @file elementwise.cpp
 * @brief Skaler eleman cekirdekleri ve ISA dagitimi.
 */

#include "matrix_pro/core/device.hpp"
#include "matrix_pro/cpu/gemm.hpp"

#include <algorithm>

namespace matrix_pro {

void add_avx2(const float* a, const float* b, float* out, int n);
void mul_avx2(const float* a, const float* b, float* out, int n);
void scale_avx2(float* data, int n, float scale);
void relu_avx2(const float* x, float* out, int n);
void maximum_avx2(float* dst, const float* src, int n);
void add_avx512(const float* a, const float* b, float* out, int n);
void mul_avx512(const float* a, const float* b, float* out, int n);
void scale_avx512(float* data, int n, float scale);
void relu_avx512(const float* x, float* out, int n);
void maximum_avx512(float* dst, const float* src, int n);

namespace {

void add_scalar(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
void mul_scalar(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}
void scale_scalar(float* data, int n, float scale) {
    for (int i = 0; i < n; ++i) {
        data[i] *= scale;
    }
}
void relu_scalar(const float* x, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}
void maximum_scalar(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = std::max(dst[i], src[i]);
    }
}

enum class Simd { Scalar, Avx2, Avx512 };

Simd simd_kind() {
    static const Simd kind = [] {
        const CpuIsa isa = detect_cpu_isa();
        if (isa.avx512f) {
            return Simd::Avx512;
        }
        if (isa.avx2) {
            return Simd::Avx2;
        }
        return Simd::Scalar;
    }();
    return kind;
}

}  // namespace

void elementwise_add(const float* a, const float* b, float* out, int n) {
    const Simd kind = simd_kind();
    if (kind == Simd::Avx512) {
        add_avx512(a, b, out, n);
    } else if (kind == Simd::Avx2) {
        add_avx2(a, b, out, n);
    } else {
        add_scalar(a, b, out, n);
    }
}
void elementwise_mul(const float* a, const float* b, float* out, int n) {
    const Simd kind = simd_kind();
    if (kind == Simd::Avx512) {
        mul_avx512(a, b, out, n);
    } else if (kind == Simd::Avx2) {
        mul_avx2(a, b, out, n);
    } else {
        mul_scalar(a, b, out, n);
    }
}
void elementwise_scale(float* data, int n, float scale) {
    const Simd kind = simd_kind();
    if (kind == Simd::Avx512) {
        scale_avx512(data, n, scale);
    } else if (kind == Simd::Avx2) {
        scale_avx2(data, n, scale);
    } else {
        scale_scalar(data, n, scale);
    }
}
void elementwise_relu(const float* x, float* out, int n) {
    const Simd kind = simd_kind();
    if (kind == Simd::Avx512) {
        relu_avx512(x, out, n);
    } else if (kind == Simd::Avx2) {
        relu_avx2(x, out, n);
    } else {
        relu_scalar(x, out, n);
    }
}
void elementwise_maximum(float* dst, const float* src, int n) {
    const Simd kind = simd_kind();
    if (kind == Simd::Avx512) {
        maximum_avx512(dst, src, n);
    } else if (kind == Simd::Avx2) {
        maximum_avx2(dst, src, n);
    } else {
        maximum_scalar(dst, src, n);
    }
}

}  // namespace matrix_pro

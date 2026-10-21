#include "matrix_pro/cpu/gemm.hpp"

#include <algorithm>
#include <immintrin.h>

namespace matrix_pro {
namespace {

int aligned_tail(int n, int width) { return n - (n % width); }

}  // namespace

void add_avx512(const float* a, const float* b, float* out, int n) {
    const int tail = aligned_tail(n, 16);
    for (int i = 0; i < tail; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    for (int i = tail; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
void mul_avx512(const float* a, const float* b, float* out, int n) {
    const int tail = aligned_tail(n, 16);
    for (int i = 0; i < tail; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
    for (int i = tail; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}
void scale_avx512(float* data, int n, float scale) {
    const __m512 factor = _mm512_set1_ps(scale);
    const int tail = aligned_tail(n, 16);
    for (int i = 0; i < tail; i += 16) {
        _mm512_storeu_ps(data + i, _mm512_mul_ps(factor, _mm512_loadu_ps(data + i)));
    }
    for (int i = tail; i < n; ++i) {
        data[i] *= scale;
    }
}
void relu_avx512(const float* x, float* out, int n) {
    const int tail = aligned_tail(n, 16);
    for (int i = 0; i < tail; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_max_ps(_mm512_setzero_ps(), _mm512_loadu_ps(x + i)));
    }
    for (int i = tail; i < n; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}
void maximum_avx512(float* dst, const float* src, int n) {
    const int tail = aligned_tail(n, 16);
    for (int i = 0; i < tail; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_max_ps(_mm512_loadu_ps(dst + i), _mm512_loadu_ps(src + i)));
    }
    for (int i = tail; i < n; ++i) {
        dst[i] = std::max(dst[i], src[i]);
    }
}

}  // namespace matrix_pro

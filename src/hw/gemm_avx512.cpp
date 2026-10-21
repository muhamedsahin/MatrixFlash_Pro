/**
 * @file gemm_avx512.cpp
 * @brief 8x16 ZMM mikrocekirdegi. AVX2 cekirdegi ile ayni paket duzeni.
 */

#include "matrix_pro/cpu/gemm.hpp"

#include <algorithm>
#include <immintrin.h>
#include <vector>

namespace matrix_pro {
namespace {

constexpr int kMr = 8;
constexpr int kNr = 16;
constexpr int kKc = 256;

void kernel_8x16(int k, const float* a, const float* b, float* c, int ldc, int n_store, float beta) {
    __m512 acc[kMr];
    for (int row = 0; row < kMr; ++row) {
        if (beta == 0.0f || n_store != kNr) {
            alignas(64) float tmp[kNr] = {};
            if (beta != 0.0f) {
                for (int col = 0; col < n_store; ++col) {
                    tmp[col] = beta * c[row * ldc + col];
                }
            }
            acc[row] = _mm512_load_ps(tmp);
        } else {
            acc[row] = _mm512_mul_ps(_mm512_set1_ps(beta), _mm512_loadu_ps(c + row * ldc));
        }
    }
    for (int p = 0; p < k; ++p) {
        const __m512 bvec = _mm512_loadu_ps(b + static_cast<std::size_t>(p) * kNr);
        if (p + 4 < k) {
            _mm_prefetch(reinterpret_cast<const char*>(b + static_cast<std::size_t>(p + 4) * kNr), _MM_HINT_T0);
        }
        for (int row = 0; row < kMr; ++row) {
            const __m512 ar = _mm512_set1_ps(a[static_cast<std::size_t>(p) * kMr + row]);
            acc[row] = _mm512_fmadd_ps(ar, bvec, acc[row]);
        }
    }
    for (int row = 0; row < kMr; ++row) {
        if (n_store == kNr) {
            _mm512_storeu_ps(c + row * ldc, acc[row]);
        } else {
            alignas(64) float tmp[kNr];
            _mm512_store_ps(tmp, acc[row]);
            for (int col = 0; col < n_store; ++col) {
                c[row * ldc + col] = tmp[col];
            }
        }
    }
}

void pack_a(float* dst, const float* a, int lda, int rows, int k, float alpha) {
    for (int p = 0; p < k; ++p) {
        for (int row = 0; row < kMr; ++row) {
            const float value = row < rows ? a[static_cast<std::size_t>(row) * static_cast<std::size_t>(lda) + p] : 0.0f;
            dst[static_cast<std::size_t>(p) * kMr + row] = alpha * value;
        }
    }
}

void pack_b(float* dst, const float* b, int ldb, int cols, int k) {
    for (int p = 0; p < k; ++p) {
        const float* brow = b + static_cast<std::size_t>(p) * static_cast<std::size_t>(ldb);
        float* dst_row = dst + static_cast<std::size_t>(p) * kNr;
        int col = 0;
        for (; col < cols; ++col) {
            dst_row[col] = brow[col];
        }
        for (; col < kNr; ++col) {
            dst_row[col] = 0.0f;
        }
    }
}

}  // namespace

void gemm_avx512_serial(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                        float alpha, float beta) {
    std::vector<float> packed_a(static_cast<std::size_t>(kKc) * kMr);
    std::vector<float> packed_b(static_cast<std::size_t>(kKc) * kNr);
    for (int p0 = 0; p0 < k; p0 += kKc) {
        const int kc = std::min(kKc, k - p0);
        const float panel_beta = p0 == 0 ? beta : 1.0f;
        for (int j0 = 0; j0 < n; j0 += kNr) {
            const int nr = std::min(kNr, n - j0);
            pack_b(packed_b.data(), b + static_cast<std::size_t>(p0) * static_cast<std::size_t>(ldb) + j0, ldb, nr, kc);
            for (int i0 = 0; i0 < m; i0 += kMr) {
                const int mr = std::min(kMr, m - i0);
                pack_a(packed_a.data(), a + static_cast<std::size_t>(i0) * static_cast<std::size_t>(lda) + p0, lda, mr, kc, alpha);
                if (mr == kMr) {
                    kernel_8x16(kc, packed_a.data(), packed_b.data(),
                                c + static_cast<std::size_t>(i0) * static_cast<std::size_t>(ldc) + j0, ldc, nr, panel_beta);
                } else {
                    for (int row = 0; row < mr; ++row) {
                        float* crow = c + static_cast<std::size_t>(i0 + row) * static_cast<std::size_t>(ldc) + j0;
                        if (panel_beta == 0.0f) {
                            for (int col = 0; col < nr; ++col) {
                                crow[col] = 0.0f;
                            }
                        } else if (panel_beta != 1.0f) {
                            for (int col = 0; col < nr; ++col) {
                                crow[col] *= panel_beta;
                            }
                        }
                        for (int p = 0; p < kc; ++p) {
                            const float coefficient = packed_a[static_cast<std::size_t>(p) * kMr + row];
                            const float* brow = packed_b.data() + static_cast<std::size_t>(p) * kNr;
                            for (int col = 0; col < nr; ++col) {
                                crow[col] += coefficient * brow[col];
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace matrix_pro

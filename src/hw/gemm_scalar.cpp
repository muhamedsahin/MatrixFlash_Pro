/**
 * @file gemm_scalar.cpp
 * @brief Referans GEMM. AVX cekirdekleri buna karsi dogrulanir.
 */

#include "matrix_pro/cpu/gemm.hpp"

#include <cstddef>

namespace matrix_pro {

void gemm_reference(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                    float alpha, float beta) {
    for (int i = 0; i < m; ++i) {
        float* crow = c + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldc);
        if (beta == 0.0f) {
            for (int j = 0; j < n; ++j) {
                crow[j] = 0.0f;
            }
        } else if (beta != 1.0f) {
            for (int j = 0; j < n; ++j) {
                crow[j] *= beta;
            }
        }
        const float* arow = a + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda);
        for (int p = 0; p < k; ++p) {
            const float coefficient = alpha * arow[p];
            const float* brow = b + static_cast<std::size_t>(p) * static_cast<std::size_t>(ldb);
            for (int j = 0; j < n; ++j) {
                crow[j] += coefficient * brow[j];
            }
        }
    }
}

}  // namespace matrix_pro

#pragma once

/**
 * @file gemm.hpp
 * @brief Satir-major tek hassasiyet matris carpimi.
 *
 * C = alpha * A * B + beta * C.
 * A m x k (adim lda), B k x n (adim ldb), C m x n (adim ldc).
 * `gemm_f32` AVX2 FMA (4x16, B paneli bir kez paketlenir) ya da skaler cekirdegi secer.
 * Seri AVX-512 cekirdegi satirlari dagitmadigi icin AVX2 varken secilmez.
 * `gemm_reference` olcum icin skaler ve belirleyicidir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

namespace matrix_pro {

void gemm_reference(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                    float alpha, float beta);

void gemm_f32(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, float alpha,
              float beta);

/// Secilen cekirdegin adi. Bu makinede AVX2 4x16 beklenir. Seri AVX-512
/// satirlari kendisi dagitmadigi icin AVX2 varken secilmez.
[[nodiscard]] const char* gemm_kernel_name();

void elementwise_add(const float* a, const float* b, float* out, int n);
void elementwise_mul(const float* a, const float* b, float* out, int n);
void elementwise_scale(float* data, int n, float scale);
void elementwise_relu(const float* x, float* out, int n);
void elementwise_maximum(float* dst, const float* src, int n);

}  // namespace matrix_pro

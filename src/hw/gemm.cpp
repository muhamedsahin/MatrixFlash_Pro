/**
 * @file gemm.cpp
 * @brief ISA secimi. AVX cekirdekleri B panelini bir kez paketleyip satirlari kendileri dagitir.
 */

#include "matrix_pro/core/device.hpp"
#include "matrix_pro/core/thread_pool.hpp"
#include "matrix_pro/cpu/gemm.hpp"

namespace matrix_pro {

void gemm_avx2_serial(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                      float alpha, float beta);
void gemm_avx512_serial(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                        float alpha, float beta);

namespace {

using GemmFn = void (*)(int, int, int, const float*, int, const float*, int, float*, int, float, float);

GemmFn selected_gemm() {
    static const GemmFn chosen = [] {
        const CpuIsa isa = detect_cpu_isa();
        // gemm_avx512_serial satirlari dagitmaz ve her N karosunda A'yi yeniden
        // paketler. Kendi satirlarini dagitan bir cekirdek olana kadar AVX2
        // daha hizlidir; bayrak tek basina secim gerekcesi degildir.
        constexpr bool kAvx512DistributesRows = false;
        if (isa.avx512f && kAvx512DistributesRows) {
            return &gemm_avx512_serial;
        }
        if (isa.avx2 && isa.fma) {
            return &gemm_avx2_serial;
        }
        return &gemm_reference;
    }();
    return chosen;
}

struct RowJob {
    GemmFn fn = nullptr;
    int n = 0;
    int k = 0;
    const float* a = nullptr;
    int lda = 0;
    const float* b = nullptr;
    int ldb = 0;
    float* c = nullptr;
    int ldc = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
};

void run_rows(int begin, int end, void* user) {
    const RowJob* job = static_cast<const RowJob*>(user);
    job->fn(end - begin, job->n, job->k, job->a + static_cast<std::size_t>(begin) * static_cast<std::size_t>(job->lda),
            job->lda, job->b, job->ldb, job->c + static_cast<std::size_t>(begin) * static_cast<std::size_t>(job->ldc),
            job->ldc, job->alpha, job->beta);
}

}  // namespace

const char* gemm_kernel_name() {
    const GemmFn fn = selected_gemm();
    if (fn == &gemm_avx2_serial) {
        return "avx2-4x16";
    }
    if (fn == &gemm_avx512_serial) {
        return "avx512-8x16";
    }
    return "reference";
}

void gemm_f32(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc, float alpha,
              float beta) {
    if (m <= 0 || n <= 0) {
        return;
    }
    GemmFn fn = selected_gemm();
    if (k <= 0) {
        gemm_reference(m, n, 0, a, lda, b, ldb, c, ldc, alpha, beta);
        return;
    }
    // AVX cekirdegi paneli bir kez paketler. Disaridan satir bolmek ayni B'yi her
    // karoda yeniden paketler. Skaler yol buyuk M'de havuzu kullanmaya devam eder.
    if (fn != &gemm_reference || m < 64) {
        fn(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
        return;
    }
    RowJob job;
    job.fn = fn;
    job.n = n;
    job.k = k;
    job.a = a;
    job.lda = lda;
    job.b = b;
    job.ldb = ldb;
    job.c = c;
    job.ldc = ldc;
    job.alpha = alpha;
    job.beta = beta;
    WorkStealingPool::instance().parallel_for(0, m, 32, &run_rows, &job);
}

}  // namespace matrix_pro

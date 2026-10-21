/**
 * @file gemm_avx2.cpp
 * @brief AVX2 FMA icin 4x16 mikrocekirdek.
 *
 * Win64'te MSVC 12 YMM biriktiriciyi dongu icinde yigina doker.
 * 4x16 (8 biriktirici) yazmaclarda kalir ve FMA kapisina oturur.
 * 6x16 ve sutun-dilimli L2 denendi. Ikisi de N=1024 ve N=2048'i yavaslatti. Secilen yol 4x16.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/thread_pool.hpp"
#include "matrix_pro/cpu/gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace matrix_pro {
namespace {

constexpr int kMr = 4;
constexpr int kNr = 16;
constexpr int kKc = 512;
// 64 satirlik A, 16 sutunluk B ile L2'de kalir. B, 4 satirda bir degil 64 satirda bir okunur.
constexpr int kMc = 64;

struct AlignedPanel {
    float* data = nullptr;
    std::size_t count = 0;

    void ensure(std::size_t needed) {
        if (count >= needed) {
            return;
        }
        release();
        count = needed;
#if defined(_MSC_VER)
        data = static_cast<float*>(_aligned_malloc(needed * sizeof(float), 64));
#else
        data = static_cast<float*>(aligned_alloc(64, ((needed * sizeof(float)) + 63u) & ~std::size_t{63}));
#endif
    }

    void release() noexcept {
        if (data == nullptr) {
            return;
        }
#if defined(_MSC_VER)
        _aligned_free(data);
#else
        free(data);
#endif
        data = nullptr;
        count = 0;
    }

    ~AlignedPanel() { release(); }
    AlignedPanel() = default;
    AlignedPanel(const AlignedPanel&) = delete;
    AlignedPanel& operator=(const AlignedPanel&) = delete;
};

thread_local AlignedPanel t_panel_a;
thread_local AlignedPanel t_panel_b;

#if !defined(_MSC_VER)
void kernel_4x16(int kc, const float* a, const float* b, float* c, int ldc, float beta) {
    __m256 c00;
    __m256 c01;
    __m256 c10;
    __m256 c11;
    __m256 c20;
    __m256 c21;
    __m256 c30;
    __m256 c31;
    if (beta == 0.0f) {
        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();
        c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();
        c31 = _mm256_setzero_ps();
    } else {
        const __m256 scale = _mm256_set1_ps(beta);
        c00 = _mm256_mul_ps(scale, _mm256_loadu_ps(c));
        c01 = _mm256_mul_ps(scale, _mm256_loadu_ps(c + 8));
        c10 = _mm256_mul_ps(scale, _mm256_loadu_ps(c + ldc));
        c11 = _mm256_mul_ps(scale, _mm256_loadu_ps(c + ldc + 8));
        c20 = _mm256_mul_ps(scale, _mm256_loadu_ps(c + 2 * ldc));
        c21 = _mm256_mul_ps(scale, _mm256_loadu_ps(c + 2 * ldc + 8));
        c30 = _mm256_mul_ps(scale, _mm256_loadu_ps(c + 3 * ldc));
        c31 = _mm256_mul_ps(scale, _mm256_loadu_ps(c + 3 * ldc + 8));
    }
    int p = 0;
    for (; p + 3 < kc; p += 4) {
        const __m256 b0 = _mm256_load_ps(b);
        const __m256 b1 = _mm256_load_ps(b + 8);
        const __m256 a0 = _mm256_broadcast_ss(a);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        const __m256 a1 = _mm256_broadcast_ss(a + 1);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        const __m256 a2 = _mm256_broadcast_ss(a + 2);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        const __m256 a3 = _mm256_broadcast_ss(a + 3);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        const __m256 b2 = _mm256_load_ps(b + 16);
        const __m256 b3 = _mm256_load_ps(b + 24);
        const __m256 a4 = _mm256_broadcast_ss(a + 4);
        c00 = _mm256_fmadd_ps(a4, b2, c00);
        c01 = _mm256_fmadd_ps(a4, b3, c01);
        const __m256 a5 = _mm256_broadcast_ss(a + 5);
        c10 = _mm256_fmadd_ps(a5, b2, c10);
        c11 = _mm256_fmadd_ps(a5, b3, c11);
        const __m256 a6 = _mm256_broadcast_ss(a + 6);
        c20 = _mm256_fmadd_ps(a6, b2, c20);
        c21 = _mm256_fmadd_ps(a6, b3, c21);
        const __m256 a7 = _mm256_broadcast_ss(a + 7);
        c30 = _mm256_fmadd_ps(a7, b2, c30);
        c31 = _mm256_fmadd_ps(a7, b3, c31);
        const __m256 b4 = _mm256_load_ps(b + 32);
        const __m256 b5 = _mm256_load_ps(b + 40);
        const __m256 a8 = _mm256_broadcast_ss(a + 8);
        c00 = _mm256_fmadd_ps(a8, b4, c00);
        c01 = _mm256_fmadd_ps(a8, b5, c01);
        const __m256 a9 = _mm256_broadcast_ss(a + 9);
        c10 = _mm256_fmadd_ps(a9, b4, c10);
        c11 = _mm256_fmadd_ps(a9, b5, c11);
        const __m256 a10 = _mm256_broadcast_ss(a + 10);
        c20 = _mm256_fmadd_ps(a10, b4, c20);
        c21 = _mm256_fmadd_ps(a10, b5, c21);
        const __m256 a11 = _mm256_broadcast_ss(a + 11);
        c30 = _mm256_fmadd_ps(a11, b4, c30);
        c31 = _mm256_fmadd_ps(a11, b5, c31);
        const __m256 b6 = _mm256_load_ps(b + 48);
        const __m256 b7 = _mm256_load_ps(b + 56);
        const __m256 a12 = _mm256_broadcast_ss(a + 12);
        c00 = _mm256_fmadd_ps(a12, b6, c00);
        c01 = _mm256_fmadd_ps(a12, b7, c01);
        const __m256 a13 = _mm256_broadcast_ss(a + 13);
        c10 = _mm256_fmadd_ps(a13, b6, c10);
        c11 = _mm256_fmadd_ps(a13, b7, c11);
        const __m256 a14 = _mm256_broadcast_ss(a + 14);
        c20 = _mm256_fmadd_ps(a14, b6, c20);
        c21 = _mm256_fmadd_ps(a14, b7, c21);
        const __m256 a15 = _mm256_broadcast_ss(a + 15);
        c30 = _mm256_fmadd_ps(a15, b6, c30);
        c31 = _mm256_fmadd_ps(a15, b7, c31);
        a += 16;
        b += 64;
        _mm_prefetch(reinterpret_cast<const char*>(b + 128), _MM_HINT_T0);
    }
    for (; p < kc; ++p) {
        const __m256 b0 = _mm256_load_ps(b);
        const __m256 b1 = _mm256_load_ps(b + 8);
        const __m256 a0 = _mm256_broadcast_ss(a);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        const __m256 a1 = _mm256_broadcast_ss(a + 1);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        const __m256 a2 = _mm256_broadcast_ss(a + 2);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        const __m256 a3 = _mm256_broadcast_ss(a + 3);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        a += kMr;
        b += kNr;
    }
    _mm256_storeu_ps(c, c00);
    _mm256_storeu_ps(c + 8, c01);
    _mm256_storeu_ps(c + ldc, c10);
    _mm256_storeu_ps(c + ldc + 8, c11);
    _mm256_storeu_ps(c + 2 * ldc, c20);
    _mm256_storeu_ps(c + 2 * ldc + 8, c21);
    _mm256_storeu_ps(c + 3 * ldc, c30);
    _mm256_storeu_ps(c + 3 * ldc + 8, c31);
}
#endif

#if defined(_MSC_VER)
extern "C" void nexus_avx2_kernel_4x16(int kc, const float* a, const float* b, float* c, int ldc, float beta);
#endif

void kernel_edge(int mr, int nr, int kc, const float* a, int a_stride, const float* b, float* c, int ldc, float beta) {
    for (int row = 0; row < mr; ++row) {
        float* crow = c + static_cast<std::size_t>(row) * static_cast<std::size_t>(ldc);
        if (beta == 0.0f) {
            int col = 0;
            for (; col + 8 <= nr; col += 8) {
                _mm256_storeu_ps(crow + col, _mm256_setzero_ps());
            }
            for (; col < nr; ++col) {
                crow[col] = 0.0f;
            }
        } else if (beta != 1.0f) {
            const __m256 scale = _mm256_set1_ps(beta);
            int col = 0;
            for (; col + 8 <= nr; col += 8) {
                _mm256_storeu_ps(crow + col, _mm256_mul_ps(scale, _mm256_loadu_ps(crow + col)));
            }
            for (; col < nr; ++col) {
                crow[col] *= beta;
            }
        }
        for (int p = 0; p < kc; ++p) {
            const __m256 coefficient = _mm256_set1_ps(a[static_cast<std::size_t>(p) * static_cast<std::size_t>(a_stride) + static_cast<std::size_t>(row)]);
            const float* brow = b + static_cast<std::size_t>(p) * kNr;
            int col = 0;
            for (; col + 8 <= nr; col += 8) {
                _mm256_storeu_ps(crow + col, _mm256_fmadd_ps(coefficient, _mm256_loadu_ps(brow + col), _mm256_loadu_ps(crow + col)));
            }
            const float scalar = a[static_cast<std::size_t>(p) * static_cast<std::size_t>(a_stride) + static_cast<std::size_t>(row)];
            for (; col < nr; ++col) {
                crow[col] += scalar * brow[col];
            }
        }
    }
}

void pack_a(float* dst, const float* a, int lda, int mr, int kc, float alpha) {
    if (mr == kMr) {
        const __m256 scale = _mm256_set1_ps(alpha);
        const float* row0 = a;
        const float* row1 = a + lda;
        const float* row2 = a + 2 * lda;
        const float* row3 = a + 3 * lda;
        int p = 0;
        for (; p + 8 <= kc; p += 8) {
            __m256 r0 = _mm256_loadu_ps(row0 + p);
            __m256 r1 = _mm256_loadu_ps(row1 + p);
            __m256 r2 = _mm256_loadu_ps(row2 + p);
            __m256 r3 = _mm256_loadu_ps(row3 + p);
            if (alpha != 1.0f) {
                r0 = _mm256_mul_ps(r0, scale);
                r1 = _mm256_mul_ps(r1, scale);
                r2 = _mm256_mul_ps(r2, scale);
                r3 = _mm256_mul_ps(r3, scale);
            }
            const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
            const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
            const __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
            const __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            float* out = dst + static_cast<std::size_t>(p) * kMr;
            _mm256_storeu_ps(out, _mm256_permute2f128_ps(u0, u1, 0x20));
            _mm256_storeu_ps(out + 8, _mm256_permute2f128_ps(u2, u3, 0x20));
            _mm256_storeu_ps(out + 16, _mm256_permute2f128_ps(u0, u1, 0x31));
            _mm256_storeu_ps(out + 24, _mm256_permute2f128_ps(u2, u3, 0x31));
        }
        for (; p < kc; ++p) {
            float* out = dst + static_cast<std::size_t>(p) * kMr;
            out[0] = alpha * row0[p];
            out[1] = alpha * row1[p];
            out[2] = alpha * row2[p];
            out[3] = alpha * row3[p];
        }
        return;
    }
    for (int p = 0; p < kc; ++p) {
        float* out = dst + static_cast<std::size_t>(p) * kMr;
        int row = 0;
        for (; row < mr; ++row) {
            out[row] = alpha * a[static_cast<std::size_t>(row) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(p)];
        }
        for (; row < kMr; ++row) {
            out[row] = 0.0f;
        }
    }
}

void pack_b(float* dst, const float* b, int ldb, int n, int kc) {
    int j = 0;
    for (; j + kNr <= n; j += kNr) {
        float* panel = dst + static_cast<std::size_t>(j / kNr) * static_cast<std::size_t>(kc) * kNr;
        for (int p = 0; p < kc; ++p) {
            const float* src = b + static_cast<std::size_t>(p) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(j);
            float* out = panel + static_cast<std::size_t>(p) * kNr;
            _mm256_store_ps(out, _mm256_loadu_ps(src));
            _mm256_store_ps(out + 8, _mm256_loadu_ps(src + 8));
        }
    }
    if (j < n) {
        const int nr = n - j;
        float* panel = dst + static_cast<std::size_t>(j / kNr) * static_cast<std::size_t>(kc) * kNr;
        for (int p = 0; p < kc; ++p) {
            const float* src = b + static_cast<std::size_t>(p) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(j);
            float* out = panel + static_cast<std::size_t>(p) * kNr;
            int col = 0;
            for (; col < nr; ++col) {
                out[col] = src[col];
            }
            for (; col < kNr; ++col) {
                out[col] = 0.0f;
            }
        }
    }
}

struct PackJob {
    const float* b = nullptr;
    int ldb = 0;
    int kc = 0;
    float* dst = nullptr;
};

void pack_columns(int begin, int end, void* user) {
    const PackJob* job = static_cast<const PackJob*>(user);
    pack_b(job->dst + static_cast<std::size_t>(begin / kNr) * static_cast<std::size_t>(job->kc) * kNr,
           job->b + static_cast<std::size_t>(begin), job->ldb, end - begin, job->kc);
}

struct PanelJob {
    const float* a = nullptr;
    int lda = 0;
    const float* packed_b = nullptr;
    float* c = nullptr;
    int ldc = 0;
    int n = 0;
    int kc = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
};

void compute_rows_small(const PanelJob* job, int begin, int end) {
    t_panel_a.ensure(static_cast<std::size_t>(kMr) * static_cast<std::size_t>(job->kc));
    float* packed_a = t_panel_a.data;
    for (int i0 = begin; i0 < end; i0 += kMr) {
        const int mr = std::min(kMr, end - i0);
        pack_a(packed_a, job->a + static_cast<std::size_t>(i0) * static_cast<std::size_t>(job->lda), job->lda, mr, job->kc, job->alpha);
        for (int j0 = 0; j0 < job->n; j0 += kNr) {
            const int nr = std::min(kNr, job->n - j0);
            const float* packed_b = job->packed_b + static_cast<std::size_t>(j0 / kNr) * static_cast<std::size_t>(job->kc) * kNr;
            float* tile = job->c + static_cast<std::size_t>(i0) * static_cast<std::size_t>(job->ldc) + static_cast<std::size_t>(j0);
            if (mr == kMr && nr == kNr) {
#if defined(_MSC_VER)
                nexus_avx2_kernel_4x16(job->kc, packed_a, packed_b, tile, job->ldc, job->beta);
#else
                kernel_4x16(job->kc, packed_a, packed_b, tile, job->ldc, job->beta);
#endif
            } else {
                kernel_edge(mr, nr, job->kc, packed_a, kMr, packed_b, tile, job->ldc, job->beta);
            }
        }
    }
}

void compute_rows(int begin, int end, void* user) {
    const PanelJob* job = static_cast<const PanelJob*>(user);
    // Kucuk N'de B zaten L2'ye sigar. 64 satirlik blok orada yalnizca paketleme ekler.
    if (job->n < 512) {
        compute_rows_small(job, begin, end);
        return;
    }
    t_panel_a.ensure(static_cast<std::size_t>(kMc) * static_cast<std::size_t>(job->kc));
    float* packed = t_panel_a.data;
    for (int block = begin; block < end; block += kMc) {
        const int rows = std::min(kMc, end - block);
        const int groups = (rows + kMr - 1) / kMr;
        for (int group = 0; group < groups; ++group) {
            const int i0 = block + group * kMr;
            const int mr = std::min(kMr, block + rows - i0);
            pack_a(packed + static_cast<std::size_t>(group) * static_cast<std::size_t>(job->kc) * kMr,
                   job->a + static_cast<std::size_t>(i0) * static_cast<std::size_t>(job->lda), job->lda, mr, job->kc, job->alpha);
        }
        for (int j0 = 0; j0 < job->n; j0 += kNr) {
            const int nr = std::min(kNr, job->n - j0);
            const float* packed_b =
                job->packed_b + static_cast<std::size_t>(j0 / kNr) * static_cast<std::size_t>(job->kc) * kNr;
            for (int group = 0; group < groups; ++group) {
                const int i0 = block + group * kMr;
                const int mr = std::min(kMr, block + rows - i0);
                const float* packed_a = packed + static_cast<std::size_t>(group) * static_cast<std::size_t>(job->kc) * kMr;
                float* tile = job->c + static_cast<std::size_t>(i0) * static_cast<std::size_t>(job->ldc) + static_cast<std::size_t>(j0);
                if (mr == kMr && nr == kNr) {
#if defined(_MSC_VER)
                    nexus_avx2_kernel_4x16(job->kc, packed_a, packed_b, tile, job->ldc, job->beta);
#else
                    kernel_4x16(job->kc, packed_a, packed_b, tile, job->ldc, job->beta);
#endif
                } else {
                    kernel_edge(mr, nr, job->kc, packed_a, kMr, packed_b, tile, job->ldc, job->beta);
                }
            }
        }
    }
}

}  // namespace

void gemm_avx2_serial(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                      float alpha, float beta) {
    const int tiles = (n + kNr - 1) / kNr;
    t_panel_b.ensure(static_cast<std::size_t>(kKc) * static_cast<std::size_t>(tiles) * kNr);
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const bool parallel = m >= 64 && flops >= 2.0e7;
    // Zen 3'te 256-bit FMA cekirdek basina ortaktir. 8 karo, 16 hyperthread'ten hizlidir.
    // B, K diliminde bir kez paketlenir. N'yi Nc'ye bolmek A'yi her dilimde yeniden
    // paketliyordu; N=1024 ve N=2048'de bu, L2 kazancindan pahaliydi.
    int grain = m / 8;
    if (grain < 32) {
        grain = 32;
    }
    grain = ((grain + kMr - 1) / kMr) * kMr;
    for (int p0 = 0; p0 < k; p0 += kKc) {
        const int kc = std::min(kKc, k - p0);
        const float* b_panel = b + static_cast<std::size_t>(p0) * static_cast<std::size_t>(ldb);
        if (parallel && n >= 256) {
            PackJob pack;
            pack.b = b_panel;
            pack.ldb = ldb;
            pack.kc = kc;
            pack.dst = t_panel_b.data;
            WorkStealingPool::instance().parallel_for(0, n, kNr * 4, &pack_columns, &pack);
        } else {
            pack_b(t_panel_b.data, b_panel, ldb, n, kc);
        }
        PanelJob job;
        job.a = a + static_cast<std::size_t>(p0);
        job.lda = lda;
        job.packed_b = t_panel_b.data;
        job.c = c;
        job.ldc = ldc;
        job.n = n;
        job.kc = kc;
        job.alpha = alpha;
        job.beta = p0 == 0 ? beta : 1.0f;
        if (parallel) {
            WorkStealingPool::instance().parallel_for(0, m, grain, &compute_rows, &job);
        } else {
            compute_rows(0, m, &job);
        }
    }
}

}  // namespace matrix_pro

#pragma once

/**
 * @file heterogeneous_scheduler.hpp
 * @brief Ayni anda CPU AVX ve CUDA GEMM.
 *
 * Satirlar ikiye bolunur. GPU payi bir CUDA akisinda, CPU payi kalici bir
 * yardimci thread ve is havuzunda ayni anda kosar. Birlestirme, satir
 * araliklari ayrık oldugu icin kopyasizdir: her taraf kendi C satirlarina yazar.
 * B'nin host baytlari onceki cagriyla ayniysa cihaz kopyasi atlanir.
 *
 * Bolme, olculen GFLOPS ve PCIe maliyetine gore yapilir. Her iki cihaz da
 * varken en az bir karo CPU'da, bir karo GPU'da kalir; boylece akis bos durmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/device.hpp"

#include <cstddef>
#include <cstdint>

namespace matrix_pro {

struct GemmSplit {
    int cpu_rows = 0;
    int gpu_rows = 0;
    int gpu_device = 0;
    double cpu_gflops = 0;
    double gpu_gflops = 0;
    double pcie_gbps = 0;
};

struct PipelineStage {
    DeviceId device{};
    /// `data` bu mikro partinin ilk satiridir. `row_stride` satir uzunlugudur.
    void (*run)(int microbatch, float* data, int row_stride, int rows, void* user) = nullptr;
    void* user = nullptr;
};

class HeterogeneousScheduler {
public:
    static HeterogeneousScheduler& instance();

    void calibrate();
    [[nodiscard]] GemmSplit plan(int m, int n, int k) const;
    [[nodiscard]] GemmSplit last_split() const;

    void gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k,
              float alpha = 1.0f, float beta = 0.0f);

    /// Asamalar mikro partiler boyunca ortusur. Asama s, parti b, onceki asamayi bekler.
    void run_pipeline(const PipelineStage* stages, int stage_count, float* data, int row_stride, int rows,
                      int microbatches);

    ~HeterogeneousScheduler();

private:
    HeterogeneousScheduler() = default;
    GemmSplit last_{};
    double cpu_gflops_ = 0;
    double gpu_gflops_ = 0;
    double pcie_gbps_ = 0;
    bool calibrated_ = false;
    float* workspace_a_ = nullptr;
    float* workspace_b_ = nullptr;
    float* workspace_c_ = nullptr;
    std::size_t cap_a_ = 0;
    std::size_t cap_b_ = 0;
    std::size_t cap_c_ = 0;
    const float* weight_host_ = nullptr;
    std::size_t weight_count_ = 0;
    std::uint64_t weight_hash_ = 0;
    struct CpuSide;
    CpuSide* cpu_side_ = nullptr;

    float* scratch(float*& slot, std::size_t& cap, std::size_t count);
    void release_scratch();
    void ensure_cpu_side();
};

/// Desteklenen HLO: tek bir f32 dot. Baska op hata firlatir.
class HloExecutable {
public:
    void run(const float* a, const float* b, float* c) const;
    int m() const { return m_; }
    int k() const { return k_; }
    int n() const { return n_; }

private:
    friend HloExecutable compile_hlo(const char* text);
    int m_ = 0;
    int k_ = 0;
    int n_ = 0;
};

[[nodiscard]] HloExecutable compile_hlo(const char* text);

}  // namespace matrix_pro

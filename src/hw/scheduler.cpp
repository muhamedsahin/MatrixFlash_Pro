#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/core/heterogeneous_scheduler.hpp"
#include "matrix_pro/cpu/gemm.hpp"
#include "matrix_pro/detail/cuda_hooks.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace matrix_pro {
namespace {

std::uint64_t hash_floats(const float* data, std::size_t count) {
    std::uint64_t hash = 0xcbf29ce484222325ull ^ count;
    std::size_t index = 0;
    while (index + 2 <= count) {
        std::uint64_t word = 0;
        std::memcpy(&word, data + index, sizeof(word));
        hash ^= word;
        hash *= 0x100000001b3ull;
        index += 2;
    }
    if (index < count) {
        std::uint32_t tail = 0;
        std::memcpy(&tail, data + index, sizeof(tail));
        hash ^= tail;
        hash *= 0x100000001b3ull;
    }
    return hash;
}

double measure_cpu_gflops() {
    constexpr int kN = 512;
    const std::size_t cells = static_cast<std::size_t>(kN) * kN;
    std::vector<float> a(cells, 0.02f);
    std::vector<float> b(cells, 0.02f);
    std::vector<float> c(cells, 0.0f);
    gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    const auto start = std::chrono::steady_clock::now();
    constexpr int kRepeats = 2;
    for (int i = 0; i < kRepeats; ++i) {
        gemm_f32(kN, kN, kN, a.data(), kN, b.data(), kN, c.data(), kN, 1.0f, 0.0f);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double flops = static_cast<double>(kRepeats) * 2.0 * kN * kN * kN;
    return seconds > 0.0 ? flops / seconds / 1.0e9 : 1.0;
}

}  // namespace

struct HeterogeneousScheduler::CpuSide {
    struct Job {
        const float* a = nullptr;
        int lda = 0;
        const float* b = nullptr;
        int ldb = 0;
        float* c = nullptr;
        int ldc = 0;
        int rows = 0;
        int n = 0;
        int k = 0;
        float alpha = 1.0f;
        float beta = 0.0f;
        std::atomic<bool>* done = nullptr;
    };

    std::mutex mutex;
    std::condition_variable ready;
    Job* job = nullptr;
    bool stop = false;
    std::thread thread;

    void loop() {
        for (;;) {
            Job* current = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] { return stop || job != nullptr; });
                if (job == nullptr && stop) {
                    return;
                }
                current = job;
                job = nullptr;
            }
            gemm_f32(current->rows, current->n, current->k, current->a, current->lda, current->b, current->ldb, current->c,
                     current->ldc, current->alpha, current->beta);
            current->done->store(true, std::memory_order_release);
        }
    }

    void start() { thread = std::thread([this] { loop(); }); }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        ready.notify_one();
        if (thread.joinable()) {
            thread.join();
        }
    }

    void submit(Job* next) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            job = next;
        }
        ready.notify_one();
    }
};

HeterogeneousScheduler& HeterogeneousScheduler::instance() {
    static HeterogeneousScheduler scheduler;
    return scheduler;
}

HeterogeneousScheduler::~HeterogeneousScheduler() {
    if (cpu_side_ != nullptr) {
        cpu_side_->shutdown();
        delete cpu_side_;
        cpu_side_ = nullptr;
    }
    release_scratch();
}

void HeterogeneousScheduler::ensure_cpu_side() {
    if (cpu_side_ != nullptr) {
        return;
    }
    cpu_side_ = new CpuSide();
    cpu_side_->start();
}

float* HeterogeneousScheduler::scratch(float*& slot, std::size_t& cap, std::size_t count) {
    if (cap < count) {
        if (slot != nullptr) {
            hw::cuda_free(0, slot);
        }
        slot = hw::cuda_alloc(0, count);
        cap = count;
    }
    return slot;
}

void HeterogeneousScheduler::release_scratch() {
    if (hw::cuda_device_count() <= 0) {
        return;
    }
    if (workspace_a_ != nullptr) {
        hw::cuda_free(0, workspace_a_);
    }
    if (workspace_b_ != nullptr) {
        hw::cuda_free(0, workspace_b_);
    }
    if (workspace_c_ != nullptr) {
        hw::cuda_free(0, workspace_c_);
    }
    workspace_a_ = workspace_b_ = workspace_c_ = nullptr;
    cap_a_ = cap_b_ = cap_c_ = 0;
}

void HeterogeneousScheduler::calibrate() {
    if (calibrated_) {
        return;
    }
    cpu_gflops_ = measure_cpu_gflops();
    if (hw::cuda_device_count() > 0) {
        gpu_gflops_ = hw::cuda_measure_gflops(0);
        pcie_gbps_ = hw::cuda_measure_pcie_gbps(0);
    }
    calibrated_ = true;
}

GemmSplit HeterogeneousScheduler::plan(int m, int n, int k) const {
    GemmSplit split;
    split.cpu_gflops = cpu_gflops_;
    split.gpu_gflops = gpu_gflops_;
    split.pcie_gbps = pcie_gbps_;
    if (m <= 0) {
        return split;
    }
    if (hw::cuda_device_count() <= 0 || gpu_gflops_ <= 0.0 || pcie_gbps_ <= 0.0) {
        split.cpu_rows = m;
        return split;
    }
    const double flops = 2.0 * static_cast<double>(m) * n * k;
    const double cpu_seconds = flops / (cpu_gflops_ * 1.0e9);
    const double gpu_seconds = flops / (gpu_gflops_ * 1.0e9);
    const double pcie = pcie_gbps_ * 1.0e9;
    const double bytes_b = static_cast<double>(k) * n * sizeof(float);
    const double bytes_per_row = static_cast<double>(k + n) * sizeof(float);
    const double fixed_transfer = bytes_b / pcie;
    const double row_transfer = (static_cast<double>(m) * bytes_per_row) / pcie;
    const double numerator = cpu_seconds - fixed_transfer;
    const double denominator = gpu_seconds + row_transfer + cpu_seconds;
    double fraction = 1.0;
    if (numerator <= 0.0 || denominator <= 0.0) {
        fraction = 0.0;
    } else {
        fraction = numerator / denominator;
    }
    if (fraction < 0.0) {
        fraction = 0.0;
    }
    if (fraction > 1.0) {
        fraction = 1.0;
    }
    int gpu_rows = static_cast<int>(fraction * static_cast<double>(m));
    if (m >= 32) {
        if (gpu_rows < 8) {
            gpu_rows = 8;
        }
        if (m - gpu_rows < 8) {
            gpu_rows = m - 8;
        }
    }
    if (gpu_rows < 0) {
        gpu_rows = 0;
    }
    if (gpu_rows > m) {
        gpu_rows = m;
    }
    split.gpu_rows = gpu_rows;
    split.cpu_rows = m - gpu_rows;
    return split;
}

GemmSplit HeterogeneousScheduler::last_split() const { return last_; }

void HeterogeneousScheduler::gemm(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n,
                                  int k, float alpha, float beta) {
    if (!calibrated_) {
        calibrate();
    }
    const bool packed = lda == k && ldb == n && ldc == n;
    GemmSplit split = plan(m, n, k);
    if (!packed || split.gpu_rows == 0 || k <= 0) {
        split.gpu_rows = 0;
        split.cpu_rows = m;
        last_ = split;
        gemm_f32(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
        return;
    }
    last_ = split;
    const int gpu_rows = split.gpu_rows;
    const int cpu_rows = split.cpu_rows;
    const std::size_t b_count = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
    const std::size_t a_count = static_cast<std::size_t>(gpu_rows) * static_cast<std::size_t>(k);
    const std::size_t c_count = static_cast<std::size_t>(gpu_rows) * static_cast<std::size_t>(n);
    const bool b_grew = cap_b_ < b_count;
    float* device_b = scratch(workspace_b_, cap_b_, b_count);
    float* device_a = scratch(workspace_a_, cap_a_, a_count);
    float* device_c = scratch(workspace_c_, cap_c_, c_count);
    if (b_grew) {
        weight_host_ = nullptr;
        weight_hash_ = 0;
    }

    std::atomic<bool> cpu_done{cpu_rows == 0};
    CpuSide::Job cpu_job;
    if (cpu_rows > 0) {
        ensure_cpu_side();
        cpu_job.a = a + static_cast<std::size_t>(gpu_rows) * static_cast<std::size_t>(lda);
        cpu_job.lda = lda;
        cpu_job.b = b;
        cpu_job.ldb = ldb;
        cpu_job.c = c + static_cast<std::size_t>(gpu_rows) * static_cast<std::size_t>(ldc);
        cpu_job.ldc = ldc;
        cpu_job.rows = cpu_rows;
        cpu_job.n = n;
        cpu_job.k = k;
        cpu_job.alpha = alpha;
        cpu_job.beta = beta;
        cpu_job.done = &cpu_done;
        cpu_side_->submit(&cpu_job);
    }
    struct WaitCpu {
        std::atomic<bool>* done = nullptr;
        ~WaitCpu() {
            if (done == nullptr) {
                return;
            }
            while (!done->load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
    } wait_cpu{cpu_rows > 0 ? &cpu_done : nullptr};

    const std::uint64_t hash = hash_floats(b, b_count);
    const bool weight_resident = !b_grew && weight_host_ == b && weight_count_ == b_count && weight_hash_ == hash;
    if (!weight_resident) {
        hw::cuda_copy_h2d(0, device_b, b, b_count);
        weight_host_ = b;
        weight_count_ = b_count;
        weight_hash_ = hash;
    }
    hw::cuda_copy_h2d(0, device_a, a, a_count);
    if (beta != 0.0f) {
        hw::cuda_copy_h2d(0, device_c, c, c_count);
    }
    hw::cuda_gemm(0, gpu_rows, n, k, device_a, k, device_b, n, device_c, n, alpha, beta);
    hw::cuda_copy_d2h(0, c, device_c, c_count);
    hw::cuda_sync(0);
}

void HeterogeneousScheduler::run_pipeline(const PipelineStage* stages, int stage_count, float* data, int row_stride,
                                          int rows, int microbatches) {
    if (stages == nullptr || stage_count <= 0 || microbatches <= 0 || rows <= 0) {
        throw MatrixProError("pipeline: empty schedule");
    }
    std::vector<std::unique_ptr<std::atomic<int>>> finished;
    finished.reserve(static_cast<std::size_t>(stage_count));
    for (int stage = 0; stage < stage_count; ++stage) {
        finished.emplace_back(new std::atomic<int>(0));
        (void)stage;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(stage_count));
    for (int stage = 0; stage < stage_count; ++stage) {
        workers.emplace_back([&, stage] {
            for (int batch = 0; batch < microbatches; ++batch) {
                if (stage > 0) {
                    while (finished[static_cast<std::size_t>(stage - 1)]->load(std::memory_order_acquire) <= batch) {
                        std::this_thread::yield();
                    }
                }
                float* view = data + (static_cast<std::size_t>(batch) * static_cast<std::size_t>(rows) *
                                      static_cast<std::size_t>(row_stride));
                stages[stage].run(batch, view, row_stride, rows, stages[stage].user);
                finished[static_cast<std::size_t>(stage)]->fetch_add(1, std::memory_order_release);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

}  // namespace matrix_pro

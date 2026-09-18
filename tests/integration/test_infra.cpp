#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/memory_mode.hpp"
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/tensor.hpp"

using matrix_pro::Matrix;
using matrix_pro::MemoryMode;
using matrix_pro::Tensor;

#include "test_support.hpp"
using matrix_pro::test::check;

// Set by a worker thread if one of its CUDA calls fails. The test context is
// intentionally not shared across threads, so the worker only flips this flag
// and main() reports it after joining.
static std::atomic<bool> thread_failure{false};

static void thread_worker(int iterations, float* result) {
    float total = 0.0f;
    try {
        for (int i = 0; i < iterations; ++i) {
            Matrix a = Matrix::ones(24, 24);
            Matrix b = Matrix::identity(24);
            Matrix c = a * b;              // cuBLAS on this thread's own stream
            total += c.sum();
        }
    } catch (const std::exception& e) {
        std::cout << "THREAD EXCEPTION: " << e.what() << "\n";
        thread_failure.store(true);
    }
    if (result != nullptr) *result = total;
}

int main() {
    try {
        const int devices = matrix_pro::device_count();
        check(devices >= 1, "at least one CUDA device is present");
        const int start_device = matrix_pro::current_device();
        check(start_device >= 0 && start_device < devices, "current device is in range");

        // DeviceScope restores the previous device on scope exit.
        {
            matrix_pro::DeviceScope scope(start_device);
            check(matrix_pro::current_device() == start_device, "DeviceScope selects the requested device");
            check(scope.previous_device() == start_device, "DeviceScope reports the previous device");
        }
        check(matrix_pro::current_device() == start_device, "DeviceScope restores the previous device");

        // An invalid device index is rejected with InvalidArgumentError.
        {
            bool caught = false;
            try { matrix_pro::set_device(devices + 7); } catch (const matrix_pro::InvalidArgumentError&) { caught = true; }
            check(caught, "set_device rejects an out-of-range index");
        }

        // Device-only mode: no host mirror until download().
        {
            Matrix d(3, 2, MemoryMode::device_only);
            check(d.device_only(), "device-only matrix reports its mode");
            check(!d.host_materialized(), "device-only matrix has no host mirror yet");
            check(d.data().empty(), "device-only data() is empty before download");

            d.fill(2.5f);                  // device-side fill
            Matrix r = d + d;              // elementwise add on device
            r.download();
            check(std::abs(r.at(0, 0) - 5.0f) < 1e-4f, "device-only arithmetic is correct");

            d.download();                  // materialize the host mirror on demand
            check(d.host_materialized(), "download() materializes the host mirror");
            check(std::abs(d.at(2, 1) - 2.5f) < 1e-4f, "downloaded device-only values are correct");

            // Host access before materializing must throw, not return garbage.
            Matrix fresh(2, 2, MemoryMode::device_only);
            fresh.fill(1.0f);
            bool caught = false;
            try { (void)fresh.at(0, 0); } catch (const matrix_pro::InvalidArgumentError&) { caught = true; }
            check(caught, "at() on an unmaterialized device-only matrix throws");
        }

        // Stale-mirror contract: host_and_device results of GPU ops refuse to
        // hand out unread host data; download() refreshes it.
        {
            Matrix a = Matrix::ones(2, 2);
            Matrix c = a + a;              // GPU kernel wrote device only
            check(c.host_current() == false, "GPU result marks its host mirror stale");
            bool caught = false;
            try { (void)c.at(0, 0); } catch (const matrix_pro::InvalidArgumentError&) { caught = true; }
            check(caught, "at() on a stale GPU result throws instead of returning zeros");
            caught = false;
            const Matrix& c_const = c;
            try { (void)c_const.data(); } catch (const matrix_pro::InvalidArgumentError&) { caught = true; }
            check(caught, "const data() on a stale GPU result throws");
            c.download();
            check(c.host_current() && std::abs(c.at(0, 0) - 2.0f) < 1e-6f, "download() refreshes the stale mirror");

            // Shape ops follow the same policy on both axes.
            Matrix r = c.reshape(4, 1);
            check(!r.host_current(), "reshape keeps the mirror stale");
            Matrix stacked = matrix_pro::concat({a, a}, 1);
            check(!stacked.host_current(), "concat (both axes) keeps the mirror stale");
            stacked.download();
            check(stacked.rows() == 2 && stacked.cols() == 4 && stacked.at(0, 3) == 1.0f,
                  "concat axis=1 values are correct after download");
        }

        // Device-only tensor fill + download.
        {
            Tensor t({2, 3}, MemoryMode::device_only);
            check(t.device_only(), "device-only tensor reports its mode");
            t.fill(4.0f);
            t.download();
            check(t.host_materialized(), "tensor download materializes the host mirror");
            check(std::abs(t.data()[5] - 4.0f) < 1e-4f, "device-only tensor values are correct");
        }

        // Thread-safety smoke test: independent threads must not corrupt results.
        {
            const int thread_count = 4;
            const int iterations = 30;
            std::vector<std::thread> threads;
            std::vector<float> results(thread_count, 0.0f);
            for (int i = 0; i < thread_count; ++i) {
                threads.emplace_back(thread_worker, iterations, &results[i]);
            }
            for (auto& thread : threads) thread.join();
            check(!thread_failure.load(), "worker threads finished without CUDA errors");
            const float expected = static_cast<float>(iterations) * 24.0f * 24.0f;
            for (int i = 0; i < thread_count; ++i) {
                check(std::abs(results[i] - expected) < 1e-2f, "multi-threaded matmul sum is correct");
            }
        }

        return matrix_pro::test::summary("INFRA");
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 99;
    }
}
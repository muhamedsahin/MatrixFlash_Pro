#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix_pro/matrix.hpp"

namespace {

struct BenchmarkResult {
    std::size_t size = 0;
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double gflops = 0.0;
    double bandwidth_gbps = 0.0;
    double relative_to_baseline = 1.0;
    double delta_vs_baseline_pct = 0.0;
};

std::vector<std::size_t> parse_sizes(const std::string& input) {
    std::vector<std::size_t> sizes;
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        const std::size_t value = static_cast<std::size_t>(std::stoull(token));
        if (value == 0) {
            throw std::invalid_argument("matrix size cannot be zero");
        }
        sizes.push_back(value);
    }
    if (sizes.empty()) {
        return {128, 256, 512, 1024};
    }
    return sizes;
}

int parse_repeats(const std::string& value, int fallback) {
    if (value.empty()) return fallback;
    try {
        const int parsed = std::stoi(value);
        return std::max(1, parsed);
    } catch (const std::exception&) {
        // BUG FIX (was): std::atoi on a non-numeric token like "--repeats" silently
        // returned 0, which collapsed to repeats=1 with no warning. Now we fail loudly
        // instead of pretending the value was valid.
        throw std::invalid_argument("--repeats expects a numeric value, got: " + value);
    }
}

double compute_operations(std::size_t size) {
    return 2.0 * static_cast<double>(size) * static_cast<double>(size) * static_cast<double>(size);
}

// BUG FIX: the old formula divided by an extra 1.0e6 and multiplied by 8.0 (a bits
// conversion that doesn't belong in a byte-based GB/s figure). Net effect was a result
// scaled down by roughly 10^12, which always rounded to 0.00 in the printed report.
//
// Correct derivation:
//   elapsed_s = elapsed_ms / 1000
//   GB/s      = (bytes / elapsed_s) / 1e9
//             = bytes / elapsed_ms * 1e-6
double compute_bandwidth_gbps(std::size_t size, double elapsed_ms) {
    if (elapsed_ms <= 0.0) return 0.0;
    // Three matrices touched by a GEMM workload: read A, read B, write C.
    const double bytes = 3.0 * static_cast<double>(size) * static_cast<double>(size) * sizeof(float);
    const double elapsed_s = elapsed_ms / 1000.0;
    return (bytes / elapsed_s) / 1.0e9;
}

// BUG FIX: previously there was no warm-up call, so the very first cuBLAS invocation
// in the whole process — which lazily creates the cuBLAS handle and can take on the
// order of 50-150ms — landed squarely inside the timed region of the first benchmarked
// size, producing a wildly inflated (and physically nonsensical) result for it.
void warm_up() {
    matrix_pro::Matrix a = matrix_pro::Matrix::ones(64, 64);
    matrix_pro::Matrix b = matrix_pro::Matrix::ones(64, 64);
    matrix_pro::Matrix c = a * b;
    c.download();
}

BenchmarkResult benchmark_size(std::size_t size, int repeats) {
    using clock = std::chrono::high_resolution_clock;

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<std::size_t>(repeats));

    for (int i = 0; i < repeats; ++i) {
        matrix_pro::Matrix left = matrix_pro::Matrix::ones(size, size);
        matrix_pro::Matrix right = matrix_pro::Matrix::ones(size, size);

        const auto start = clock::now();
        matrix_pro::Matrix result = left * right;
        result.download();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();
        samples_ms.push_back(elapsed_ms);

        if (result.at(0, 0) <= 0.0f) {
            throw std::runtime_error("benchmark result validation failed");
        }
    }

    const double avg_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / static_cast<double>(samples_ms.size());
    const double min_ms = *std::min_element(samples_ms.begin(), samples_ms.end());
    const double max_ms = *std::max_element(samples_ms.begin(), samples_ms.end());
    const double ops = compute_operations(size);
    const double gflops = ops / (avg_ms * 1.0e6);
    const double bandwidth_gbps = compute_bandwidth_gbps(size, avg_ms);

    return BenchmarkResult{size, avg_ms, min_ms, max_ms, gflops, bandwidth_gbps, 1.0, 0.0};
}

void print_report(std::vector<BenchmarkResult>& results, std::size_t baseline_size, int repeats) {
    const auto baseline_it = std::find_if(results.begin(), results.end(), [&](const BenchmarkResult& r) {
        return r.size == baseline_size;
    });

    if (baseline_it == results.end()) {
        throw std::runtime_error("baseline size not found in benchmark results");
    }

    const double baseline_gflops = baseline_it->gflops;

    std::cout << "\n========================================\n";
    std::cout << " MatrixFlash-Pro GPU Benchmark Report\n";
    std::cout << "========================================\n";
    std::cout << "Repeats per size: " << repeats << " (warm-up run excluded from all measurements)\n\n";
    std::cout << std::left << std::setw(12) << "Size"
              << std::setw(12) << "Avg(ms)"
              << std::setw(12) << "Min(ms)"
              << std::setw(12) << "Max(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(16) << "BW(GB/s)"
              << std::setw(16) << "Rel. to base"
              << "\n";

    std::cout << std::setfill('-') << std::setw(90) << "" << std::setfill(' ') << "\n";

    for (auto& result : results) {
        result.relative_to_baseline = baseline_gflops > 0.0 ? result.gflops / baseline_gflops : 1.0;
        result.delta_vs_baseline_pct = (result.relative_to_baseline - 1.0) * 100.0;

        std::cout << std::left << std::setw(12) << result.size
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.avg_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.min_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.max_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops
                  << std::setw(16) << std::fixed << std::setprecision(2) << result.bandwidth_gbps
                  << std::setw(16) << std::fixed << std::setprecision(2) << result.relative_to_baseline << "x"
                  << "\n";
    }

    std::cout << "\nInterpretation:\n";
    std::cout << "- Baseline size: " << baseline_size << "x" << baseline_size << " (reference)\n";
    std::cout << "- Relative to baseline > 1.0x indicates faster throughput than the reference size.\n";
    std::cout << "- GFLOPS is computed as: 2 * n^3 / elapsed_time_seconds\n";
    std::cout << "- Memory bandwidth is estimated from the matrix data movement involved in the multiply workload.\n";

    const auto best = std::max_element(results.begin(), results.end(), [](const BenchmarkResult& a, const BenchmarkResult& b) {
        return a.gflops < b.gflops;
    });

    std::cout << "\nBest throughput observed: " << best->size << "x" << best->size
              << " with " << std::fixed << std::setprecision(2) << best->gflops << " GFLOPS\n";
}

} // namespace

int main(int argc, char** argv) {
    // BUG FIX: the previous parser only accepted ONE of --sizes / --repeats as the
    // first argument and treated everything else positionally. Passing both flags
    // together (e.g. "--sizes 128,256 --repeats 3") silently fed the literal string
    // "--repeats" into parse_repeats, which used to fall back to 1 without warning.
    // This version scans all arguments as independent --flag value pairs, in any order.
    std::vector<std::size_t> sizes;
    int repeats = 5;
    bool sizes_set = false;
    bool repeats_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--sizes" || arg == "-s") && i + 1 < argc) {
            sizes = parse_sizes(argv[++i]);
            sizes_set = true;
        } else if ((arg == "--repeats" || arg == "-r") && i + 1 < argc) {
            repeats = parse_repeats(argv[++i], repeats);
            repeats_set = true;
        } else if (!sizes_set) {
            // Backwards-compatible fallback: a bare positional list of sizes.
            sizes = parse_sizes(arg);
            sizes_set = true;
        } else if (!repeats_set) {
            // Backwards-compatible fallback: a bare positional repeat count.
            repeats = parse_repeats(arg, repeats);
            repeats_set = true;
        }
    }

    if (sizes.empty()) {
        sizes = {128, 256, 512, 1024};
    }

    std::sort(sizes.begin(), sizes.end());

    // BUG FIX: run one untimed multiply first so cuBLAS handle creation / CUDA
    // context warm-up cost never leaks into the first measured size.
    warm_up();

    std::vector<BenchmarkResult> results;
    results.reserve(sizes.size());

    for (std::size_t size : sizes) {
        results.push_back(benchmark_size(size, repeats));
    }

    const std::size_t baseline_size = sizes.front();
    print_report(results, baseline_size, repeats);
    return 0;
}
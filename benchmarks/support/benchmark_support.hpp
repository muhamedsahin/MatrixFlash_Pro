#pragma once

// ============================================================================
//  MatrixFlash-Pro benchmark harness
//  ---------------------------------------------------------------------------
//  One measurement engine shared by every benchmark program:
//
//    * CUDA-event timing (device time only -- no host-side copy noise)
//    * warm-up calls excluded from the statistics
//    * repeated samples reported as mean / median / min / max / stddev
//    * throughput helpers for GFLOPS and GB/s figures
//    * CSV / JSON export for regression tracking
//
//  A workload family lives in one benchmarks/bench_<name>.cpp file and exposes
//  a single factory function returning its BenchmarkInfo. Registration is
//  explicit (see all_benchmarks()) on purpose: relying on static initializers
//  inside a static library is unreliable, because the linker is free to drop
//  object files that nothing references.
// ============================================================================

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace matrix_pro {
namespace bench {

// --- command line -----------------------------------------------------------
struct Options {
    // Size list interpreted per workload (matrix edge, element count, image
    // size, ...). Empty means "use the workload's own default list".
    std::vector<std::size_t> sizes;
    int repeats = 5;          // measured samples per case
    int warmup = 2;           // untimed calls before measuring
    bool quiet = false;       // suppress per-case progress lines
    std::vector<std::string> only;   // substring filter on the workload name
    std::string csv_path;
    std::string json_path;
};

// --- statistics -------------------------------------------------------------
struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

// --- one measured case ------------------------------------------------------
struct Result {
    std::string bench;      // workload family ("matmul", "elementwise", ...)
    std::string label;      // human readable case ("1024x1024", "16M elems")
    std::size_t size = 0;   // primary dimension of the case
    Stats ms;               // device time per call
    double throughput = 0.0;
    std::string unit;       // "GFLOPS", "GB/s", "samples/s", empty = n/a
    std::string note;       // free-form annotation (accuracy, speedup, ...)
};

using BenchFn = void (*)(const Options& options, std::vector<Result>& out);

struct BenchmarkInfo {
    std::string name;
    std::string summary;
    BenchFn run = nullptr;
};

// --- workload registry ------------------------------------------------------
// Each benchmarks/bench_*.cpp implements exactly one of these factories.
const BenchmarkInfo& matmul_benchmark();
const BenchmarkInfo& elementwise_benchmark();
const BenchmarkInfo& reductions_benchmark();
const BenchmarkInfo& activations_benchmark();
const BenchmarkInfo& linalg_benchmark();
const BenchmarkInfo& conv_benchmark();
const BenchmarkInfo& precision_benchmark();
const BenchmarkInfo& sparse_benchmark();
const BenchmarkInfo& training_benchmark();
const BenchmarkInfo& comparison_benchmark();
const BenchmarkInfo& external_gemm_benchmark();
const BenchmarkInfo& external_linalg_benchmark();
const BenchmarkInfo& external_sparse_benchmark();

// Every known workload, in run order.
std::vector<BenchmarkInfo> all_benchmarks();
// --- measurement primitives -------------------------------------------------
// Runs `warmup` untimed calls, then `repeats` CUDA-event-timed calls and
// returns the statistics of the timed samples (device time in milliseconds).
Stats sample_ms(const std::function<void()>& fn, int warmup, int repeats);

// Same, but honours options.warmup / options.repeats.
Stats sample_ms(const std::function<void()>& fn, const Options& options);

// Throughput helpers. `flops` / `bytes` describe one call of the timed function.
double to_gflops(double flops, double ms);
double to_gbps(std::size_t bytes, double ms);

// True when `bytes` still fit comfortably (60% of the currently free device
// memory) on the active GPU. Workloads use it to skip a case instead of dying
// with an out-of-memory error when the user asks for a very large --sizes entry.
bool memory_available(std::size_t bytes);

// Appends a fully populated Result.
void add_result(std::vector<Result>& out,
                const std::string& bench,
                const std::string& label,
                std::size_t size,
                const Stats& ms,
                double throughput,
                const std::string& unit,
                const std::string& note = std::string());

// --- reporting / export -----------------------------------------------------
void print_device_banner();
void print_report(const Options& options, const std::vector<Result>& results);
void write_csv(const std::string& path, const std::vector<Result>& results);
void write_json(const std::string& path, const std::vector<Result>& results);

// --- entry points -----------------------------------------------------------
// `single_bench` is the workload name for the per-family executables, or an
// empty string for the aggregate runner (matrix_pro_bench_all).
int run_main(int argc, char** argv, const char* single_bench, const char* exe_name);

// --- small parsing helpers shared by the workload files ---------------------
// Accepts plain integers plus K/M/G suffixes (1024, 512K, 4M, 2G).
std::size_t parse_size(const std::string& text);
std::string format_size(std::size_t value);
std::string format_matrix_label(std::size_t rows, std::size_t cols);

} // namespace bench
} // namespace matrix_pro

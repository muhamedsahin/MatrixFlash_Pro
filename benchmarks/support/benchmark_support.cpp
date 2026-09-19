#include "benchmark_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "matrix_pro/core/cuda_utils.hpp"

namespace matrix_pro {
namespace bench {
namespace {

std::string escape_json(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

Stats compute_stats(std::vector<double> samples) {
    Stats stats;
    if (samples.empty()) {
        return stats;
    }
    std::sort(samples.begin(), samples.end());
    stats.min = samples.front();
    stats.max = samples.back();
    const std::size_t middle = samples.size() / 2;
    stats.median = (samples.size() % 2 == 0)
                       ? 0.5 * (samples[middle - 1] + samples[middle])
                       : samples[middle];
    double total = 0.0;
    for (double value : samples) {
        total += value;
    }
    stats.mean = total / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double value : samples) {
        variance += (value - stats.mean) * (value - stats.mean);
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

std::string format_double(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string format_integer(std::size_t value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

std::string timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_MSC_VER)
    localtime_s(&local, &now);
#else
    local = *std::localtime(&now);
#endif
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &local);
    return std::string(buffer);
}

// Pad a string to `width` (left aligned) without relying on iostream state.
std::string pad(const std::string& text, std::size_t width) {
    if (text.size() >= width) {
        return text;
    }
    return text + std::string(width - text.size(), ' ');
}

std::string throughput_cell(const Result& result) {
    if (result.unit.empty()) {
        return "-";
    }
    return format_double(result.throughput, 2) + " " + result.unit;
}

// RAII wrapper so a throwing workload can never leak its CUDA events.
class EventGuard {
public:
    explicit EventGuard(cudaEvent_t event) : event_(event) {}
    ~EventGuard() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    EventGuard(const EventGuard&) = delete;
    EventGuard& operator=(const EventGuard&) = delete;
    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

} // namespace

// ============================================================================
//  Workload registry
// ============================================================================
std::vector<BenchmarkInfo> all_benchmarks() {
    return {matmul_benchmark(),
            elementwise_benchmark(),
            reductions_benchmark(),
            activations_benchmark(),
            linalg_benchmark(),
            conv_benchmark(),
            precision_benchmark(),
            sparse_benchmark(),
            training_benchmark(),
            comparison_benchmark(),
            external_gemm_benchmark(),
            external_linalg_benchmark(),
            external_sparse_benchmark(),
            math_benchmark(),
            execution_benchmark(),
            serialization_benchmark()};
}

// ============================================================================
//  Measurement
// ============================================================================
Stats sample_ms(const std::function<void()>& fn, int warmup, int repeats) {
    const int warmup_runs = std::max(0, warmup);
    const int measured_runs = std::max(1, repeats);

    for (int i = 0; i < warmup_runs; ++i) {
        fn();
    }

    cudaStream_t stream = compute_stream();
    cudaEvent_t start_raw = nullptr;
    cudaEvent_t stop_raw = nullptr;
    checkCuda(cudaEventCreate(&start_raw), "benchmark event create");
    EventGuard start(start_raw);
    checkCuda(cudaEventCreate(&stop_raw), "benchmark event create");
    EventGuard stop(stop_raw);

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(measured_runs));
    for (int i = 0; i < measured_runs; ++i) {
        checkCuda(cudaEventRecord(start.get(), stream), "benchmark start event");
        fn();
        checkCuda(cudaEventRecord(stop.get(), stream), "benchmark stop event");
        checkCuda(cudaEventSynchronize(stop.get()), "benchmark event sync");
        float elapsed = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsed, start.get(), stop.get()),
                  "benchmark event elapsed");
        samples.push_back(static_cast<double>(elapsed));
    }
    return compute_stats(samples);
}

Stats sample_ms(const std::function<void()>& fn, const Options& options) {
    return sample_ms(fn, options.warmup, options.repeats);
}

double to_gflops(double flops, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return flops / (ms * 1.0e6);
}

double to_gbps(std::size_t bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    const double seconds = ms / 1000.0;
    return (static_cast<double>(bytes) / seconds) / 1.0e9;
}

void add_result(std::vector<Result>& out,
                const std::string& bench,
                const std::string& label,
                std::size_t size,
                const Stats& ms,
                double throughput,
                const std::string& unit,
                const std::string& note) {
    Result result;
    result.bench = bench;
    result.label = label;
    result.size = size;
    result.ms = ms;
    result.throughput = throughput;
    result.unit = unit;
    result.note = note;
    out.push_back(std::move(result));
}

bool memory_available(std::size_t bytes) {
    if (device_count() <= 0) {
        return false;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return false;
    }
    // Keep a generous headroom: kernels, workspace and the host mirror all
    // compete for the same memory.
    const double headroom = static_cast<double>(free_bytes) * 0.6;
    return static_cast<double>(bytes) <= headroom;
}

// ============================================================================
//  Device banner
// ============================================================================
void print_device_banner() {
    std::cout << "MatrixFlash-Pro benchmark harness\n";
    std::cout << "  host      : " << (sizeof(void*) == 8 ? "x64" : "x86")
              << ", CUDA toolkit headers " << CUDART_VERSION << "\n";
    std::cout << "  timestamp : " << timestamp() << "\n";

    int devices = device_count();
    std::cout << "  devices   : " << devices << "\n";
    if (devices <= 0) {
        std::cout << "  WARNING   : no CUDA device available\n";
        return;
    }

    const int index = current_device();
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, index) == cudaSuccess) {
        const double memory_gib =
            static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  gpu       : " << properties.name << " (device " << index << ")\n";
        std::cout << "  compute   : sm_" << properties.major << properties.minor
                  << ", " << properties.multiProcessorCount << " SMs, "
                  << format_double(memory_gib, 2) << " GiB\n";
    }

    int runtime_version = 0;
    int driver_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        std::cout << "  cuda rt   : " << runtime_version / 1000 << "."
                  << (runtime_version % 1000) / 10 << "\n";
    }
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        std::cout << "  driver    : " << driver_version / 1000 << "."
                  << (driver_version % 1000) / 10 << "\n";
    }
    std::cout << std::flush;
}

// ============================================================================
//  Reporting
// ============================================================================
void print_report(const Options& options, const std::vector<Result>& results) {
    std::cout << "\n==============================================================================\n";
    std::cout << " MatrixFlash-Pro benchmark summary\n";
    std::cout << "   samples per case : " << options.repeats
              << "   (untimed warm-up calls: " << options.warmup << ")\n";
    std::cout << "   timing           : CUDA events, device time per call\n";
    std::cout << "==============================================================================\n";

    std::vector<std::string> order;
    for (const Result& result : results) {
        if (std::find(order.begin(), order.end(), result.bench) == order.end()) {
            order.push_back(result.bench);
        }
    }

    // Column width follows the longest case label so wide labels such as
    // "conv2d 32->64 p1 224" never collide with the numeric columns.
    std::size_t case_width = 20;
    for (const Result& result : results) {
        case_width = std::max(case_width, result.label.size() + 2);
    }

    for (const std::string& bench : order) {
        std::cout << "\n[" << bench << "]\n";
        std::cout << pad("case", case_width) << pad("mean ms", 11) << pad("median ms", 11)
                  << pad("min ms", 11) << pad("max ms", 11) << pad("stddev", 10)
                  << pad("throughput", 16) << "note\n";
        std::cout << std::string(case_width + 60, '-') << "\n";

        const Result* best = nullptr;
        for (const Result& result : results) {
            if (result.bench != bench) {
                continue;
            }
            if (best == nullptr ||
                (result.throughput > 0.0 && result.throughput > best->throughput)) {
                best = &result;
            }
            std::cout << pad(result.label, case_width)
                      << pad(format_double(result.ms.mean, 4), 11)
                      << pad(format_double(result.ms.median, 4), 11)
                      << pad(format_double(result.ms.min, 4), 11)
                      << pad(format_double(result.ms.max, 4), 11)
                      << pad(format_double(result.ms.stddev, 4), 10)
                      << pad(throughput_cell(result), 16)
                      << result.note << "\n";
        }
        if (best != nullptr) {
            std::cout << "  best: " << best->label << " -> " << throughput_cell(*best)
                      << " (median " << format_double(best->ms.median, 4) << " ms)\n";
        }
    }
    std::cout << std::flush;
}

// ============================================================================
//  Export
// ============================================================================
void write_csv(const std::string& path, const std::vector<Result>& results) {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot open CSV output file: " + path);
    }
    file << "bench,case,size,ms_mean,ms_median,ms_min,ms_max,ms_stddev,throughput,unit,note\n";
    for (const Result& result : results) {
        file << result.bench << ',' << result.label << ',' << result.size << ','
             << format_double(result.ms.mean, 6) << ','
             << format_double(result.ms.median, 6) << ','
             << format_double(result.ms.min, 6) << ','
             << format_double(result.ms.max, 6) << ','
             << format_double(result.ms.stddev, 6) << ','
             << format_double(result.throughput, 4) << ','
             << result.unit << ',' << result.note << '\n';
    }
    std::cout << "CSV written: " << path << '\n';
}

void write_json(const std::string& path, const std::vector<Result>& results) {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot open JSON output file: " + path);
    }
    file << "{\n";
    file << "  \"library\": \"MatrixFlash-Pro\",\n";
    file << "  \"timestamp\": \"" << timestamp() << "\",\n";

    const int index = device_count() > 0 ? current_device() : 0;
    cudaDeviceProp properties{};
    if (device_count() > 0 && cudaGetDeviceProperties(&properties, index) == cudaSuccess) {
        file << "  \"device\": {\n";
        file << "    \"name\": \"" << escape_json(properties.name) << "\",\n";
        file << "    \"index\": " << index << ",\n";
        file << "    \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n";
        file << "    \"sm_count\": " << properties.multiProcessorCount << ",\n";
        file << "    \"memory_bytes\": " << properties.totalGlobalMem << "\n";
        file << "  },\n";
    } else {
        file << "  \"device\": null,\n";
    }

    file << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const Result& result = results[i];
        file << "    {\n";
        file << "      \"bench\": \"" << escape_json(result.bench) << "\",\n";
        file << "      \"case\": \"" << escape_json(result.label) << "\",\n";
        file << "      \"size\": " << result.size << ",\n";
        file << "      \"ms_mean\": " << format_double(result.ms.mean, 6) << ",\n";
        file << "      \"ms_median\": " << format_double(result.ms.median, 6) << ",\n";
        file << "      \"ms_min\": " << format_double(result.ms.min, 6) << ",\n";
        file << "      \"ms_max\": " << format_double(result.ms.max, 6) << ",\n";
        file << "      \"ms_stddev\": " << format_double(result.ms.stddev, 6) << ",\n";
        file << "      \"throughput\": " << format_double(result.throughput, 4) << ",\n";
        file << "      \"unit\": \"" << escape_json(result.unit) << "\",\n";
        file << "      \"note\": \"" << escape_json(result.note) << "\"\n";
        file << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    file << "  ]\n";
    file << "}\n";
    std::cout << "JSON written: " << path << '\n';
}

// ============================================================================
//  CLI parsing helpers
// ============================================================================
namespace {

void ensure_output_directories(const std::string& path) {
    const std::filesystem::path file_path(path);
    const std::filesystem::path parent = file_path.parent_path();
    if (parent.empty()) {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
        throw std::runtime_error("cannot create output directory '" + parent.string() +
                                 "': " + error.message());
    }
}

void print_help(const char* exe_name) {
    std::cout
        << "usage: " << exe_name << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -s, --sizes <list>    case sizes, comma separated (256,512,1024 or 1M,4M)\n"
        << "  -r, --repeats <n>     measured samples per case (default 5)\n"
        << "  -w, --warmup <n>      untimed warm-up calls per case (default 2)\n"
        << "  -f, --filter <text>   run only the workloads whose name contains <text>\n"
        << "  -q, --quiet           do not print per-workload progress lines\n"
        << "      --csv <path>      export the results table as CSV\n"
        << "      --json <path>     export the results table plus device info as JSON\n"
        << "      --list            list the available workloads and exit\n"
        << "  -h, --help            show this message\n"
        << "\n"
        << "Positional arguments are kept for backwards compatibility: the first one\n"
        << "is a size list, the second a repeat count.\n";
}

int parse_int(const std::string& text, const char* flag) {
    try {
        const int value = std::stoi(text);
        if (value < 0) {
            throw std::invalid_argument("negative");
        }
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(flag) +
                                    " expects a non-negative integer, got: " + text);
    }
}

std::vector<std::size_t> parse_size_list(const std::string& text) {
    std::vector<std::size_t> sizes;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        sizes.push_back(parse_size(token));
    }
    if (sizes.empty()) {
        throw std::invalid_argument("empty size list");
    }
    return sizes;
}

} // namespace

std::size_t parse_size(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("empty size");
    }
    std::size_t multiplier = 1;
    std::string digits = text;
    switch (text.back()) {
        case 'k': case 'K': multiplier = 1024; digits.pop_back(); break;
        case 'm': case 'M': multiplier = 1024 * 1024; digits.pop_back(); break;
        case 'g': case 'G':
            multiplier = static_cast<std::size_t>(1024) * 1024 * 1024;
            digits.pop_back();
            break;
        default: break;
    }
    if (digits.empty()) {
        throw std::invalid_argument("invalid size: " + text);
    }
    unsigned long long value = 0;
    try {
        value = std::stoull(digits);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid size: " + text);
    }
    if (value == 0) {
        throw std::invalid_argument("size must be positive: " + text);
    }
    return static_cast<std::size_t>(value) * multiplier;
}

std::string format_size(std::size_t value) {
    const std::size_t kilo = 1024;
    const std::size_t mega = kilo * 1024;
    const std::size_t giga = mega * 1024;
    if (value >= giga && value % giga == 0) {
        return format_integer(value / giga) + "G";
    }
    if (value >= mega && value % mega == 0) {
        return format_integer(value / mega) + "M";
    }
    if (value >= kilo && value % kilo == 0) {
        return format_integer(value / kilo) + "K";
    }
    return format_integer(value);
}

std::string format_matrix_label(std::size_t rows, std::size_t cols) {
    return format_integer(rows) + "x" + format_integer(cols);
}

int run_main(int argc, char** argv, const char* single_bench, const char* exe_name) {
    try {
        Options options;
        bool list_only = false;
        bool sizes_set = false;
        bool repeats_set = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_help(exe_name);
                return 0;
            }
            if (arg == "--list") {
                list_only = true;
                continue;
            }
            if (arg == "--quiet" || arg == "-q") {
                options.quiet = true;
                continue;
            }
            if ((arg == "--sizes" || arg == "-s") && i + 1 < argc) {
                options.sizes = parse_size_list(argv[++i]);
                sizes_set = true;
                continue;
            }
            if ((arg == "--repeats" || arg == "-r") && i + 1 < argc) {
                options.repeats = std::max(1, parse_int(argv[++i], "--repeats"));
                repeats_set = true;
                continue;
            }
            if ((arg == "--warmup" || arg == "-w") && i + 1 < argc) {
                options.warmup = parse_int(argv[++i], "--warmup");
                continue;
            }
            if ((arg == "--filter" || arg == "-f") && i + 1 < argc) {
                options.only.emplace_back(argv[++i]);
                continue;
            }
            if (arg == "--csv" && i + 1 < argc) {
                options.csv_path = argv[++i];
                continue;
            }
            if (arg == "--json" && i + 1 < argc) {
                options.json_path = argv[++i];
                continue;
            }
            // Backwards compatible positional arguments.
            if (!arg.empty() && arg.front() >= '0' && arg.front() <= '9') {
                if (!sizes_set) {
                    options.sizes = parse_size_list(arg);
                    sizes_set = true;
                    continue;
                }
                if (!repeats_set) {
                    options.repeats = std::max(1, parse_int(arg, "repeats"));
                    repeats_set = true;
                    continue;
                }
                throw std::invalid_argument("unexpected extra positional argument: " + arg);
            }
            throw std::invalid_argument("unknown argument: " + arg + " (try --help)");
        }

        if (single_bench != nullptr && *single_bench != '\0') {
            options.only.emplace_back(single_bench);
        }

        std::vector<BenchmarkInfo> selected;
        for (const BenchmarkInfo& info : all_benchmarks()) {
            bool keep = options.only.empty();
            for (const std::string& filter : options.only) {
                if (info.name.find(filter) != std::string::npos) {
                    keep = true;
                    break;
                }
            }
            if (keep) {
                selected.push_back(info);
            }
        }

        if (list_only) {
            std::cout << "available workloads:\n";
            for (const BenchmarkInfo& info : selected) {
                std::cout << "  " << pad(info.name, 14) << info.summary << "\n";
            }
            return 0;
        }

        if (selected.empty()) {
            throw std::runtime_error("no workload matched the requested filter");
        }

        print_device_banner();

        std::vector<Result> results;
        for (const BenchmarkInfo& info : selected) {
            if (!options.quiet) {
                std::cout << "\n>>> " << info.name << " - " << info.summary << "\n" << std::flush;
            }
            info.run(options, results);
        }

        print_report(options, results);

        if (!options.csv_path.empty()) {
            ensure_output_directories(options.csv_path);
            write_csv(options.csv_path, results);
        }
        if (!options.json_path.empty()) {
            ensure_output_directories(options.json_path);
            write_json(options.json_path, results);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << "\n";
        return 1;
    }
}

} // namespace bench
} // namespace matrix_pro

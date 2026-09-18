// cuSOLVER-backed linear algebra: solvers and decompositions.
//
// Throughput is reported in operations per second (1000 / median ms), because
// LAPACK-class routines on small matrices are latency-bound rather than
// bandwidth- or FLOP-bound.

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/advanced.hpp"
#include "matrix_pro/ops/linalg.hpp"

namespace matrix_pro {
namespace bench {
namespace {

std::vector<std::size_t> default_sizes() {
    return {64, 128, 256, 512};
}

// Symmetric positive definite: n*I + 0.5*ones (diagonally dominant).
Matrix make_spd(std::size_t n) {
    std::vector<float> values(n * n, 0.5f);
    for (std::size_t i = 0; i < n; ++i) {
        values[i * n + i] += static_cast<float>(n);
    }
    return Matrix(n, n, values);
}

// Deterministic, well-conditioned and non-symmetric.
Matrix make_general(std::size_t n) {
    std::vector<float> values(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t index = i * n + j;
            values[index] = static_cast<float>((index * 2654435761u) % 10000u) / 1000.0f + 0.1f;
        }
    }
    return Matrix(n, n, values);
}

void run(const Options& options_orig, std::vector<Result>& out) {
    Options options = options_orig;
    if (options.sizes.empty()) {
        options.sizes = default_sizes();
    }

    // name -> factory taking the problem size and performing one call.
    struct Case {
        const char* name;
        std::function<void(std::size_t)> call;
        bool needs_spd;
    };

    const std::vector<Case> cases{
        {"solve", [](std::size_t n) { (void)make_spd(n).solve(make_spd(n)); }, true},
        {"solve_least_squares", [](std::size_t n) { (void)make_general(n).solve_least_squares(make_general(n)); }, false},
        {"qr", [](std::size_t n) { (void)make_general(n).qr(); }, false},
        {"svd", [](std::size_t n) { (void)make_general(n).svd(); }, false},
        {"cholesky", [](std::size_t n) { (void)make_spd(n).cholesky(); }, true},
        {"eigen", [](std::size_t n) { (void)make_spd(n).eigen(); }, true},
        {"pinv", [](std::size_t n) { (void)make_general(n).pinv(); }, false},
        {"rank", [](std::size_t n) { (void)make_general(n).rank(); }, false},
        {"determinant", [](std::size_t n) { (void)make_general(n).determinant(); }, false},
        {"inverse", [](std::size_t n) { (void)make_general(n).inverse(); }, false},
        {"condition_number", [](std::size_t n) { (void)make_general(n).condition_number(); }, false},
    };

    for (const std::size_t n : options.sizes) {
        if (n < 8) {
            continue;
        }
        // Decompositions allocate a few n x n work buffers; keep a wide margin.
        if (!memory_available(16 * n * n * sizeof(float))) {
            if (!options.quiet) {
                std::cout << "  (skipping " << format_matrix_label(n, n)
                          << ": not enough free device memory)\n";
            }
            continue;
        }

        const std::string label = format_matrix_label(n, n);
        for (const Case& item : cases) {
            const Stats ms = sample_ms([&] { item.call(n); }, options);
            const double per_second = ms.median > 0.0 ? 1000.0 / ms.median : 0.0;
            add_result(out, "linalg", label, n, ms, per_second, "ops/s",
                       std::string(item.name) + (item.needs_spd ? " (spd)" : ""));
        }
    }
}

} // namespace

const BenchmarkInfo& linalg_benchmark() {
    static const BenchmarkInfo info{"linalg",
                                    "cuSOLVER solvers / decompositions: latency per call",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro

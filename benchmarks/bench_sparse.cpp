// Sparse (CSR) kernels against their dense counterparts.
//
// Every case uses a fixed 16 non-zeros per row, which makes the sparsity level
// independent of the matrix size and lets the table compare spmv / sparse-dense
// GEMM across sizes. Where a dense baseline fits in memory, the speedup versus
// cuBLAS is reported in the note column.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/product.hpp"
#include "matrix_pro/sparse/sparse.hpp"

namespace matrix_pro {
namespace bench {
namespace {

constexpr std::size_t kNonZerosPerRow = 16;
constexpr std::size_t kDenseColumns = 64;

std::vector<std::size_t> default_sizes() {
    return {1024, 4096, 16384};
}

SparseCSR make_sparse(std::size_t n) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;
    rows.reserve(n * kNonZerosPerRow);
    cols.reserve(n * kNonZerosPerRow);
    values.reserve(n * kNonZerosPerRow);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < kNonZerosPerRow; ++j) {
            rows.push_back(static_cast<int>(i));
            cols.push_back(static_cast<int>(((i * 7) + j * 31) % n));
            values.push_back(0.5f + 0.01f * static_cast<float>(j));
        }
    }
    return SparseCSR::from_coo(n, n, rows, cols, values);
}

std::string speedup_note(const char* kernel, double dense_ms, double sparse_ms) {
    if (sparse_ms <= 0.0) {
        return kernel;
    }
    const double factor = dense_ms / sparse_ms;
    const long long tenths = static_cast<long long>(factor * 10.0 + 0.5);
    return std::string(kernel) + ", " + std::to_string(tenths / 10) + "." +
           std::to_string(tenths % 10) + "x vs dense";
}

void run(const Options& options_orig, std::vector<Result>& out) {
    Options options = options_orig;
    if (options.sizes.empty()) {
        options.sizes = default_sizes();
    }

    for (const std::size_t n : options.sizes) {
        if (n < kNonZerosPerRow * 2) {
            continue;
        }

        const std::size_t nnz = n * kNonZerosPerRow;
        const std::size_t sparse_bytes = nnz * (sizeof(float) + sizeof(int)) + (n + 1) * sizeof(int);
        if (!memory_available(4 * sparse_bytes)) {
            if (!options.quiet) {
                std::cout << "  (skipping sparse " << n << ": not enough free device memory)\n";
            }
            continue;
        }

        const SparseCSR sparse = make_sparse(n);
        Matrix x = Matrix::ones(n, 1);

        // --- spmv ------------------------------------------------------------
        {
            Matrix reference = matrix_pro::spmv(sparse, x);
            reference.download();
            if (!std::isfinite(reference.at(0, 0))) {
                throw std::runtime_error("spmv produced a non-finite value");
            }

            const std::size_t bytes = sparse_bytes + 3 * n * sizeof(float);
            const Stats ms = sample_ms([&] {
                Matrix result = matrix_pro::spmv(sparse, x);
                (void)result;
            }, options);

            std::string note = "spmv, 16 nnz/row";
            if (n <= 4096 && memory_available(n * n * sizeof(float))) {
                Matrix dense = Matrix::ones(n, n);
                const Stats dense_ms = sample_ms([&] {
                    Matrix result = multiply(dense, x);
                    (void)result;
                }, options);
                note = speedup_note("spmv, 16 nnz/row", dense_ms.median, ms.median);
            }
            add_result(out, "sparse", std::to_string(n), n, ms,
                       to_gbps(bytes, ms.median), "GB/s", note);
        }

        // --- sparse x dense GEMM ---------------------------------------------
        {
            Matrix dense_rhs = Matrix::ones(n, kDenseColumns);
            const Stats ms = sample_ms([&] {
                Matrix result = matrix_pro::sparse_matmul(sparse, dense_rhs);
                (void)result;
            }, options);

            const double flops = 2.0 * static_cast<double>(nnz) * kDenseColumns;
            std::string note = "sparse x dense (64 cols)";
            if (n <= 4096 && memory_available(n * n * sizeof(float))) {
                Matrix dense = Matrix::ones(n, n);
                const Stats dense_ms = sample_ms([&] {
                    Matrix result = multiply(dense, dense_rhs);
                    (void)result;
                }, options);
                note = speedup_note("sparse x dense", dense_ms.median, ms.median);
            }
            add_result(out, "sparse", std::to_string(n), n, ms,
                       to_gflops(flops, ms.median), "GFLOPS", note);
        }

        // --- reconstruction cost ---------------------------------------------
        {
            const Stats ms = sample_ms([&] {
                Matrix dense = sparse.to_dense();
                (void)dense;
            }, options);
            add_result(out, "sparse", std::to_string(n) + " to_dense", n, ms,
                       to_gbps(n * n * sizeof(float), ms.median), "GB/s", "CSR -> dense");
        }
    }
}

} // namespace

const BenchmarkInfo& sparse_benchmark() {
    static const BenchmarkInfo info{"sparse",
                                    "CSR spmv / sparse-dense GEMM vs dense cuBLAS baselines",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro
// Part 1: includes. Raw cuSPARSE SpMV/SpMM vs mflash spmv/sparse_matmul.
// cuSPARSE ships with the CUDA toolkit: zero extra installs for FAZ A.
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "benchmark_support.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/sparse/sparse.hpp"

namespace matrix_pro {
namespace bench {
namespace {

void check_sp(cusparseStatus_t status, const char* op) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(op) + ": cusparse status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

constexpr std::size_t kNnzPerRow = 16;

SparseCSR make_pattern(std::size_t n) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> values;
    rows.reserve(n * kNnzPerRow);
    cols.reserve(n * kNnzPerRow);
    values.reserve(n * kNnzPerRow);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < kNnzPerRow; ++j) {
            rows.push_back(static_cast<int>(i));
            cols.push_back(static_cast<int>(((i * 7) + j * 31) % n));
            values.push_back(0.5f + 0.01f * static_cast<float>(j));
        }
    return SparseCSR::from_coo(n, n, rows, cols, values);
}

// Raw cuSPARSE SpMV on the same CSR pattern (generic API, no wrapper).
void raw_spmv_once(const SparseCSR& csr, const float* d_x, float* d_y) {
    cusparseHandle_t handle = nullptr;
    check_sp(cusparseCreate(&handle), "sp create");
    cusparseSetStream(handle, compute_stream());
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr, vec_y = nullptr;
    const std::size_t n = csr.rows();
    int64_t rows64 = static_cast<int64_t>(n);
    int64_t cols64 = static_cast<int64_t>(n);
    int64_t nnz64 = static_cast<int64_t>(csr.nnz());
    check_sp(cusparseCreateCsr(&mat, rows64, cols64, nnz64,
                               const_cast<int*>(csr.row_ptr()),
                               const_cast<int*>(csr.col_idx()),
                               const_cast<float*>(csr.values()),
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
             "sp mat create");
    check_sp(cusparseCreateDnVec(&vec_x, cols64, const_cast<float*>(d_x),
                                 CUDA_R_32F),
             "sp vecx");
    check_sp(cusparseCreateDnVec(&vec_y, rows64, d_y, CUDA_R_32F), "sp vecy");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::size_t buf_bytes = 0;
    check_sp(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, mat, vec_x, &beta, vec_y,
                                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                     &buf_bytes),
             "sp bufsize");
    void* d_buf = nullptr;
    if (buf_bytes > 0) {
        checkCuda(cudaMalloc(&d_buf, buf_bytes), "sp buf alloc");
    }
    check_sp(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat,
                           vec_x, &beta, vec_y, CUDA_R_32F,
                           CUSPARSE_SPMV_ALG_DEFAULT, d_buf),
             "sp spmv");
    if (d_buf != nullptr) cudaFree(d_buf);
    cusparseDestroyDnVec(vec_y);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroySpMat(mat);
    cusparseDestroy(handle);
}

void run_sparse_pairs(const Options& options, std::vector<Result>& out,
                      const std::vector<std::size_t>& sizes) {
    for (const std::size_t n : sizes) {
        if (n < kNnzPerRow * 2) {
            continue;
        }
        const std::size_t nnz = n * kNnzPerRow;
        const std::size_t sparse_bytes =
            nnz * (sizeof(float) + sizeof(int)) + (n + 1) * sizeof(int);
        if (!memory_available(4 * sparse_bytes)) {
            if (!options.quiet) {
                std::cout << "  (skipping external-sparse " << n << ")\n";
            }
            continue;
        }
        const SparseCSR csr = make_pattern(n);
        Matrix x = Matrix::ones(n, 1);
        const std::size_t bytes = sparse_bytes + 3 * n * sizeof(float);

        {
            const Stats ms = sample_ms(
                [&] {
                    Matrix y = matrix_pro::spmv(csr, x);
                    (void)y;
                },
                options);
            add_result(out, "external-sparse", std::to_string(n), n, ms,
                       to_gbps(bytes, ms.median), "GB/s",
                       "mflash spmv, 16 nnz/row (measured)");
        }
        {
            Matrix y(n, 1);
            const Stats ms = sample_ms(
                [&] { raw_spmv_once(csr, x.device_data(), y.device_data()); },
                options);
            checkCuda(cudaGetLastError(), "raw spmv launch");
            y.download();
            if (!std::isfinite(y.at(0, 0))) {
                throw std::runtime_error("raw cusparse spmv non-finite");
            }
            add_result(out, "external-sparse", std::to_string(n), n, ms,
                       to_gbps(bytes, ms.median), "GB/s",
                       "raw cusparse SpMV (measured)");
        }
    }
}

void run_external_sparse(const Options& options, std::vector<Result>& out) {
    const std::vector<std::size_t> sizes =
        options.sizes.empty() ? std::vector<std::size_t>{1024, 4096} : options.sizes;
    run_sparse_pairs(options, out, sizes);
}

} // namespace

const BenchmarkInfo& external_sparse_benchmark() {
    static const BenchmarkInfo info{
        "external_sparse", "raw cuSPARSE SpMV vs mflash spmv (same CSR pattern)",
        run_external_sparse};
    return info;
}

} // namespace bench
} // namespace matrix_pro


#pragma once

// Minimal CSR sparse-matrix support (embedding tables, graph models).
// Values stay on the device; row_ptr/col_idx are int32 device buffers managed
// by the class. cuSPARSE is used when available, otherwise a coalesced
// fallback kernel is used (see src/operations_sparse.cu).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "matrix_pro/core/errors.hpp"

namespace matrix_pro {

class Matrix;

class SparseCSR {
public:
    SparseCSR() = default;
    SparseCSR(std::size_t rows, std::size_t cols, std::size_t nnz);
    // Build from dense host data (zeros are dropped with |v| <= threshold).
    static SparseCSR from_dense(const Matrix& dense, float threshold = 0.0f);
    // Build from explicit COO triplets.
    static SparseCSR from_coo(std::size_t rows, std::size_t cols,
                              const std::vector<int>& row_idx,
                              const std::vector<int>& col_idx,
                              const std::vector<float>& values);

    SparseCSR(const SparseCSR& other);
    SparseCSR& operator=(const SparseCSR& other);
    SparseCSR(SparseCSR&& other) noexcept;
    SparseCSR& operator=(SparseCSR&& other) noexcept;

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t nnz() const noexcept { return nnz_; }
    bool empty() const noexcept { return nnz_ == 0; }
    float sparsity() const noexcept;

    const int* row_ptr() const noexcept { return row_ptr_.get(); }
    const int* col_idx() const noexcept { return col_idx_.get(); }
    const float* values() const noexcept { return values_.get(); }
    int* row_ptr() noexcept { return row_ptr_.get(); }
    int* col_idx() noexcept { return col_idx_.get(); }
    float* values() noexcept { return values_.get(); }

    // Copy the structure back to host (for inspection / tests).
    std::vector<int> host_row_ptr() const;
    std::vector<int> host_col_idx() const;
    std::vector<float> host_values() const;

    Matrix to_dense() const;
    void synchronize() const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t nnz_ = 0;
    std::unique_ptr<int, void (*)(int*)> row_ptr_{nullptr, nullptr};
    std::unique_ptr<int, void (*)(int*)> col_idx_{nullptr, nullptr};
    std::unique_ptr<float, void (*)(float*)> values_{nullptr, nullptr};
};

// ============================================================================
//  Sparse (CSR) dense-sparse mixed helpers.
// ============================================================================
// y = A * x  (A: m x n sparse, x: n x 1 dense -> m x 1 dense).
Matrix spmv(const SparseCSR& a, const Matrix& x);
// C = A * B (A sparse m x k, B dense k x n -> dense m x n).
Matrix sparse_matmul(const SparseCSR& a, const Matrix& b);
// Folds a dense product into an existing sparsity pattern while keeping the
// structure frozen: for every stored entry (r, c) of `acc`,
//     acc[r][c] += (a * b)[r][c]
// `acc` must be (m x n) with a (m x k) and b (k x n). This is the classic
// embedding-table gradient update: the pattern stays exactly as sparse as it
// was created, so the table never densifies. The accumulation runs on the host
// and uploads the updated values; it is meant for periodic (per-step) updates,
// not for hot inner loops.
void sparse_outer_accumulate(SparseCSR& acc, const Matrix& a, const Matrix& b);

} // namespace matrix_pro

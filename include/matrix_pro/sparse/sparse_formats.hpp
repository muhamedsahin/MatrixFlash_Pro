#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace matrix_pro {

class Matrix;
class SparseCSR;

// COO (Coordinate) sparse format
class SparseCOO {
public:
    SparseCOO() = default;
    SparseCOO(std::size_t rows, std::size_t cols, std::size_t nnz);
    ~SparseCOO();
    SparseCOO(const SparseCOO& other);
    SparseCOO& operator=(const SparseCOO& other);
    SparseCOO(SparseCOO&& other) noexcept;
    SparseCOO& operator=(SparseCOO&& other) noexcept;

    static SparseCOO from_dense(const Matrix& dense, float threshold = 0.0f);
    static SparseCOO from_triplets(std::size_t rows, std::size_t cols,
                                    const std::vector<int>& row_idx,
                                    const std::vector<int>& col_idx,
                                    const std::vector<float>& values);

    Matrix to_dense() const;
    SparseCSR to_csr() const;
    SparseCOO coalesce() const;

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t nnz() const noexcept { return nnz_; }
    float sparsity() const;

    const int* row_indices() const noexcept;
    const int* col_indices() const noexcept;
    const float* values() const noexcept;

private:
    std::size_t rows_ = 0, cols_ = 0, nnz_ = 0;
    std::unique_ptr<int, void(*)(int*)> row_idx_{nullptr, nullptr};
    std::unique_ptr<int, void(*)(int*)> col_idx_{nullptr, nullptr};
    std::unique_ptr<float, void(*)(float*)> values_{nullptr, nullptr};
};

// CSC (Compressed Sparse Column) format
class SparseCSC {
public:
    SparseCSC() = default;
    SparseCSC(std::size_t rows, std::size_t cols, std::size_t nnz);
    ~SparseCSC();
    SparseCSC(const SparseCSC& other);
    SparseCSC& operator=(const SparseCSC& other);
    SparseCSC(SparseCSC&& other) noexcept;
    SparseCSC& operator=(SparseCSC&& other) noexcept;

    static SparseCSC from_dense(const Matrix& dense, float threshold = 0.0f);
    static SparseCSC from_csr(const SparseCSR& csr);

    Matrix to_dense() const;
    SparseCSR to_csr() const;
    SparseCSC transpose() const;

    Matrix spmv(const Matrix& x) const;
    Matrix sparse_matmul(const Matrix& B) const;

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t nnz() const noexcept { return nnz_; }
    float sparsity() const;

private:
    std::size_t rows_ = 0, cols_ = 0, nnz_ = 0;
    std::unique_ptr<int, void(*)(int*)> col_ptr_{nullptr, nullptr};
    std::unique_ptr<int, void(*)(int*)> row_idx_{nullptr, nullptr};
    std::unique_ptr<float, void(*)(float*)> values_{nullptr, nullptr};
};

// Format conversion utilities
SparseCSR coo_to_csr(const SparseCOO& coo);
SparseCOO csr_to_coo(const SparseCSR& csr);
SparseCSC csr_to_csc(const SparseCSR& csr);
SparseCSR csc_to_csr(const SparseCSC& csc);

// Sparse-sparse operations
SparseCSR spgemm(const SparseCSR& A, const SparseCSR& B);
SparseCSR sparse_add(const SparseCSR& A, const SparseCSR& B, float alpha = 1.0f, float beta = 1.0f);

}

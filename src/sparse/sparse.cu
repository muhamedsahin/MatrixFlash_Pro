#include "matrix_pro/sparse/sparse.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace matrix_pro {
namespace {

void release_int(int* p) { if (p) free_device_memory(p); }
void release_float(float* p) { if (p) free_device_memory(p); }

__global__ void spmv_kernel(const int* row_ptr, const int* col_idx, const float* values,
                            const float* x, float* y, std::size_t rows) {
    const std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float sum = 0.0f;
    for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
        sum += values[j] * x[col_idx[j]];
    y[r] = sum;
}

__global__ void sparse_matmul_kernel(const int* row_ptr, const int* col_idx, const float* values,
                                     const float* b, float* c,
                                     std::size_t rows, std::size_t n) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= rows * n) return;
    const std::size_t r = t / n;
    const std::size_t col = t % n;
    float sum = 0.0f;
    for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
        sum += values[j] * b[static_cast<std::size_t>(col_idx[j]) * n + col];
    c[t] = sum;
}

} // namespace

SparseCSR::SparseCSR(std::size_t rows, std::size_t cols, std::size_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz),
      row_ptr_(nullptr, release_int), col_idx_(nullptr, release_int),
      values_(nullptr, release_float) {
    if (nnz_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        col_idx_.reset(static_cast<int*>(allocate_device_memory(nnz_ * sizeof(int))));
        values_.reset(static_cast<float*>(allocate_device_memory(nnz_ * sizeof(float))));
    } else if (rows_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        zero_device_memory(row_ptr_.get(), (rows_ + 1) * sizeof(int));
    }
}

SparseCSR SparseCSR::from_dense(const Matrix& dense, float threshold) {
    Matrix copy = dense;
    copy.download();
    const std::size_t rows = dense.rows(), cols = dense.cols();
    std::vector<int> row_ptr(rows + 1, 0);
    std::vector<int> col_idx;
    std::vector<float> values;
    col_idx.reserve(dense.size() / 8 + 1);
    values.reserve(dense.size() / 8 + 1);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const float v = copy.data()[r * cols + c];
            if (v > threshold || v < -threshold) {
                col_idx.push_back(static_cast<int>(c));
                values.push_back(v);
            }
        }
        row_ptr[r + 1] = static_cast<int>(col_idx.size());
    }
    SparseCSR out(rows, cols, values.size());
    if (!values.empty()) {
        checkCuda(cudaMemcpyAsync(out.row_ptr_.get(), row_ptr.data(), (rows + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream()), "csr upload");
        checkCuda(cudaMemcpyAsync(out.col_idx_.get(), col_idx.data(), values.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream()), "csr col upload");
        checkCuda(cudaMemcpyAsync(out.values_.get(), values.data(), values.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, compute_stream()), "csr val upload");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr upload sync");
    }
    return out;
}

SparseCSR SparseCSR::from_coo(std::size_t rows, std::size_t cols,
                              const std::vector<int>& row_idx,
                              const std::vector<int>& col_idx,
                              const std::vector<float>& values) {
    if (row_idx.size() != col_idx.size() || row_idx.size() != values.size())
        throw ShapeMismatchError("from_coo triplet size mismatch");
    std::vector<int> order(row_idx.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return row_idx[a] < row_idx[b] || (row_idx[a] == row_idx[b] && col_idx[a] < col_idx[b]);
    });
    std::vector<int> row_ptr(rows + 1, 0);
    std::vector<int> cols_sorted;
    std::vector<float> vals_sorted;
    for (int o : order) {
        if (row_idx[o] < 0 || static_cast<std::size_t>(row_idx[o]) >= rows ||
            col_idx[o] < 0 || static_cast<std::size_t>(col_idx[o]) >= cols)
            throw OutOfRangeError("from_coo index out of range");
        cols_sorted.push_back(col_idx[o]);
        vals_sorted.push_back(values[o]);
        row_ptr[static_cast<std::size_t>(row_idx[o]) + 1]++;
    }
    for (std::size_t r = 0; r < rows; ++r) row_ptr[r + 1] += row_ptr[r];
    SparseCSR out(rows, cols, values.size());
    if (!vals_sorted.empty()) {
        checkCuda(cudaMemcpyAsync(out.row_ptr_.get(), row_ptr.data(), (rows + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream()), "coo upload");
        checkCuda(cudaMemcpyAsync(out.col_idx_.get(), cols_sorted.data(), vals_sorted.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream()), "coo col upload");
        checkCuda(cudaMemcpyAsync(out.values_.get(), vals_sorted.data(), vals_sorted.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, compute_stream()), "coo val upload");
        checkCuda(cudaStreamSynchronize(compute_stream()), "coo upload sync");
    }
    return out;
}

SparseCSR::SparseCSR(const SparseCSR& other)
    : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_),
      row_ptr_(nullptr, release_int), col_idx_(nullptr, release_int),
      values_(nullptr, release_float) {
    if (nnz_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        col_idx_.reset(static_cast<int*>(allocate_device_memory(nnz_ * sizeof(int))));
        values_.reset(static_cast<float*>(allocate_device_memory(nnz_ * sizeof(float))));
        checkCuda(cudaMemcpyAsync(row_ptr_.get(), other.row_ptr_.get(), (rows_ + 1) * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()), "csr copy");
        checkCuda(cudaMemcpyAsync(col_idx_.get(), other.col_idx_.get(), nnz_ * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()), "csr copy");
        checkCuda(cudaMemcpyAsync(values_.get(), other.values_.get(), nnz_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice, compute_stream()), "csr copy");
    } else if (rows_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        checkCuda(cudaMemcpyAsync(row_ptr_.get(), other.row_ptr_.get(), (rows_ + 1) * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()), "csr copy");
    }
}

SparseCSR& SparseCSR::operator=(const SparseCSR& other) {
    if (this != &other) *this = SparseCSR(other);
    return *this;
}

SparseCSR::SparseCSR(SparseCSR&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_),
      row_ptr_(std::move(other.row_ptr_)), col_idx_(std::move(other.col_idx_)),
      values_(std::move(other.values_)) {
    other.rows_ = other.cols_ = other.nnz_ = 0;
}

SparseCSR& SparseCSR::operator=(SparseCSR&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_; cols_ = other.cols_; nnz_ = other.nnz_;
        row_ptr_ = std::move(other.row_ptr_);
        col_idx_ = std::move(other.col_idx_);
        values_ = std::move(other.values_);
        other.rows_ = other.cols_ = other.nnz_ = 0;
    }
    return *this;
}

float SparseCSR::sparsity() const noexcept {
    if (rows_ == 0 || cols_ == 0) return 0.0f;
    return 1.0f - static_cast<float>(nnz_) / static_cast<float>(rows_ * cols_);
}

std::vector<int> SparseCSR::host_row_ptr() const {
    std::vector<int> host(rows_ + 1, 0);
    if (rows_ > 0) {
        checkCuda(cudaMemcpyAsync(host.data(), row_ptr_.get(), (rows_ + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost, compute_stream()), "csr read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr sync");
    }
    return host;
}

std::vector<int> SparseCSR::host_col_idx() const {
    std::vector<int> host(nnz_, 0);
    if (nnz_ > 0) {
        checkCuda(cudaMemcpyAsync(host.data(), col_idx_.get(), nnz_ * sizeof(int),
                                  cudaMemcpyDeviceToHost, compute_stream()), "csr read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr sync");
    }
    return host;
}

std::vector<float> SparseCSR::host_values() const {
    std::vector<float> host(nnz_, 0.0f);
    if (nnz_ > 0) {
        checkCuda(cudaMemcpyAsync(host.data(), values_.get(), nnz_ * sizeof(float),
                                  cudaMemcpyDeviceToHost, compute_stream()), "csr read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr sync");
    }
    return host;
}

Matrix SparseCSR::to_dense() const {
    Matrix out(rows_, cols_);
    if (nnz_ == 0) return out;
    const std::vector<int> rp = host_row_ptr();
    const std::vector<int> ci = host_col_idx();
    const std::vector<float> vs = host_values();
    for (std::size_t r = 0; r < rows_; ++r)
        for (int j = rp[r]; j < rp[r + 1]; ++j)
            out.data()[r * cols_ + static_cast<std::size_t>(ci[j])] = vs[j];
    out.upload();
    return out;
}

void SparseCSR::synchronize() const { matrix_pro::synchronize(); }

Matrix spmv(const SparseCSR& a, const Matrix& x) {
    if (x.cols() != 1 || x.rows() != a.cols())
        throw ShapeMismatchError("spmv requires (m x n) sparse and (n x 1) dense");
    Matrix y(a.rows(), 1);
    if (a.empty()) return y;
    spmv_kernel<<<(a.rows() + 255) / 256, 256, 0, compute_stream()>>>(
        a.row_ptr(), a.col_idx(), a.values(), x.device_data(), y.device_data(), a.rows());
    checkCuda(cudaGetLastError(), "spmv kernel launch");
    y.mark_host_stale();
    return y;
}

Matrix sparse_matmul(const SparseCSR& a, const Matrix& b) {
    if (b.rows() != a.cols())
        throw ShapeMismatchError("sparse_matmul inner dimensions must match");
    Matrix c(a.rows(), b.cols());
    if (a.empty() || c.empty()) return c;
    sparse_matmul_kernel<<<(c.size() + 255) / 256, 256, 0, compute_stream()>>>(
        a.row_ptr(), a.col_idx(), a.values(), b.device_data(), c.device_data(),
        a.rows(), b.cols());
    checkCuda(cudaGetLastError(), "sparse_matmul kernel launch");
    c.mark_host_stale();
    return c;
}

void sparse_outer_accumulate(SparseCSR& acc, const Matrix& a, const Matrix& b) {
    if (a.rows() != acc.rows() || b.cols() != acc.cols())
        throw ShapeMismatchError("sparse_outer_accumulate shape mismatch");
    if (a.cols() != b.rows())
        throw ShapeMismatchError("sparse_outer_accumulate inner dimensions must match");
    if (acc.empty()) return;

    Matrix product = a * b;
    product.download();

    const std::vector<int> row_ptr = acc.host_row_ptr();
    const std::vector<int> col_idx = acc.host_col_idx();
    std::vector<float> values = acc.host_values();
    for (std::size_t row = 0; row + 1 < row_ptr.size(); ++row) {
        for (int index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
            const std::size_t col = static_cast<std::size_t>(col_idx[static_cast<std::size_t>(index)]);
            values[static_cast<std::size_t>(index)] += product.at(row, col);
        }
    }

    checkCuda(cudaMemcpy(acc.values(), values.data(), values.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "sparse_outer_accumulate value upload");
    acc.synchronize();
}

} // namespace matrix_pro



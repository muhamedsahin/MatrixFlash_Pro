#include "matrix_pro/sparse/sparse_formats.hpp"
#include "matrix_pro/sparse.hpp"
#include "matrix_pro/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/core/kernel_helpers.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <vector>
#include <algorithm>
#include <numeric>

namespace matrix_pro {

namespace {

void release_int(int* p) { if (p) free_device_memory(p); }
void release_float(float* p) { if (p) free_device_memory(p); }

cusparseHandle_t get_cusparse_handle() {
    static thread_local cusparseHandle_t handle = nullptr;
    if (!handle) {
        cusparseCreate(&handle);
    }
    return handle;
}

#define CHECK_CUSPARSE(stat) \
    do { \
        if (stat != CUSPARSE_STATUS_SUCCESS) { \
            throw matrix_pro::RuntimeError("cuSPARSE Error!"); \
        } \
    } while(0)

__global__ void coo_to_dense_kernel(const int* row_idx, const int* col_idx, const float* values,
                                     float* dense, std::size_t cols, std::size_t nnz) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        dense[row_idx[i] * cols + col_idx[i]] = values[i];
    }
}

__global__ void count_nnz_per_col_kernel(const int* col_idx, int* counts, std::size_t nnz) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&counts[col_idx[i]], 1);
    }
}

__global__ void csr_to_csc_scatter_kernel(const int* row_ptr, const int* col_idx, const float* values,
                                           int* csc_col_ptr, int* csc_row_idx, float* csc_values,
                                           int* write_pos, std::size_t rows) {
    auto r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j) {
            int col = col_idx[j];
            int pos = atomicAdd(&write_pos[col], 1);
            int dest = csc_col_ptr[col] + pos;
            csc_row_idx[dest] = r;
            csc_values[dest] = values[j];
        }
    }
}

__global__ void coo_to_csr_count_kernel(const int* row_idx, int* row_counts, std::size_t nnz) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&row_counts[row_idx[i]], 1);
    }
}

__global__ void csc_spmv_kernel(const int* col_ptr, const int* row_idx, const float* values,
                                 const float* x, float* y, std::size_t cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float x_col = x[col];
        for (int j = col_ptr[col]; j < col_ptr[col + 1]; ++j) {
            atomicAdd(&y[row_idx[j]], values[j] * x_col);
        }
    }
}

} // namespace

// SparseCOO Implementation
SparseCOO::SparseCOO(std::size_t rows, std::size_t cols, std::size_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz),
      row_idx_(static_cast<int*>(allocate_device_memory(nnz * sizeof(int))), release_int),
      col_idx_(static_cast<int*>(allocate_device_memory(nnz * sizeof(int))), release_int),
      values_(static_cast<float*>(allocate_device_memory(nnz * sizeof(float))), release_float) {}

SparseCOO::~SparseCOO() = default;

SparseCOO::SparseCOO(const SparseCOO& other) : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_) {
    if (nnz_ > 0) {
        row_idx_.reset(static_cast<int*>(allocate_device_memory(nnz_ * sizeof(int))));
        col_idx_.reset(static_cast<int*>(allocate_device_memory(nnz_ * sizeof(int))));
        values_.reset(static_cast<float*>(allocate_device_memory(nnz_ * sizeof(float))));
        checkCuda(cudaMemcpyAsync(row_idx_.get(), other.row_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream()), "memcpy");
        checkCuda(cudaMemcpyAsync(col_idx_.get(), other.col_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream()), "memcpy");
        checkCuda(cudaMemcpyAsync(values_.get(), other.values_.get(), nnz_ * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream()), "memcpy");
    }
}

SparseCOO& SparseCOO::operator=(const SparseCOO& other) {
    if (this != &other) {
        *this = SparseCOO(other);
    }
    return *this;
}

SparseCOO::SparseCOO(SparseCOO&& other) noexcept = default;
SparseCOO& SparseCOO::operator=(SparseCOO&& other) noexcept = default;

const int* SparseCOO::row_indices() const noexcept { return row_idx_.get(); }
const int* SparseCOO::col_indices() const noexcept { return col_idx_.get(); }
const float* SparseCOO::values() const noexcept { return values_.get(); }
float SparseCOO::sparsity() const { return 1.0f - static_cast<float>(nnz_) / (rows_ * cols_); }

SparseCOO SparseCOO::from_dense(const Matrix& dense, float threshold) {
    std::vector<float> host_dense(dense.rows() * dense.cols());
    checkCuda(cudaMemcpy(host_dense.data(), dense.data(), host_dense.size() * sizeof(float), cudaMemcpyDeviceToHost), "from_dense memcpy");
    
    std::vector<int> row_idx, col_idx;
    std::vector<float> values;
    
    for (std::size_t r = 0; r < dense.rows(); ++r) {
        for (std::size_t c = 0; c < dense.cols(); ++c) {
            float val = host_dense[r * dense.cols() + c];
            if (std::abs(val) > threshold) {
                row_idx.push_back(r);
                col_idx.push_back(c);
                values.push_back(val);
            }
        }
    }
    return from_triplets(dense.rows(), dense.cols(), row_idx, col_idx, values);
}

SparseCOO SparseCOO::from_triplets(std::size_t rows, std::size_t cols,
                                   const std::vector<int>& row_idx,
                                   const std::vector<int>& col_idx,
                                   const std::vector<float>& values) {
    SparseCOO coo(rows, cols, values.size());
    checkCuda(cudaMemcpy(coo.row_idx_.get(), row_idx.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy");
    checkCuda(cudaMemcpy(coo.col_idx_.get(), col_idx.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy");
    checkCuda(cudaMemcpy(coo.values_.get(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
    return coo;
}

Matrix SparseCOO::to_dense() const {
    Matrix dense(rows_, cols_);
    dense.fill_zero();
    if (nnz_ > 0) {
        coo_to_dense_kernel<<<detail::capped_grid(nnz_, 256), 256, 0, compute_stream()>>>(
            row_idx_.get(), col_idx_.get(), values_.get(), dense.data(), cols_, nnz_);
    }
    return dense;
}

SparseCSR SparseCOO::to_csr() const {
    return coo_to_csr(*this);
}

SparseCOO SparseCOO::coalesce() const {
    // For now, assume sorted or simply download, sort, merge, upload.
    // A robust implementation would use thrust::sort_by_key on device.
    // Here we download for simplicity of merging.
    std::vector<int> h_r(nnz_), h_c(nnz_);
    std::vector<float> h_v(nnz_);
    checkCuda(cudaMemcpy(h_r.data(), row_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    checkCuda(cudaMemcpy(h_c.data(), col_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    checkCuda(cudaMemcpy(h_v.data(), values_.get(), nnz_ * sizeof(float), cudaMemcpyDeviceToHost), "memcpy");
    
    struct Entry { int r, c; float v; };
    std::vector<Entry> entries(nnz_);
    for(size_t i=0; i<nnz_; ++i) entries[i] = {h_r[i], h_c[i], h_v[i]};
    
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b){
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });
    
    std::vector<int> out_r, out_c;
    std::vector<float> out_v;
    for (const auto& e : entries) {
        if (!out_r.empty() && out_r.back() == e.r && out_c.back() == e.c) {
            out_v.back() += e.v;
        } else {
            out_r.push_back(e.r);
            out_c.push_back(e.c);
            out_v.push_back(e.v);
        }
    }
    return from_triplets(rows_, cols_, out_r, out_c, out_v);
}


// SparseCSC Implementation
SparseCSC::SparseCSC(std::size_t rows, std::size_t cols, std::size_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz),
      col_ptr_(static_cast<int*>(allocate_device_memory((cols + 1) * sizeof(int))), release_int),
      row_idx_(static_cast<int*>(allocate_device_memory(nnz * sizeof(int))), release_int),
      values_(static_cast<float*>(allocate_device_memory(nnz * sizeof(float))), release_float) {}

SparseCSC::~SparseCSC() = default;

SparseCSC::SparseCSC(const SparseCSC& other) : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_) {
    if (cols_ + 1 > 0) {
        col_ptr_.reset(static_cast<int*>(allocate_device_memory((cols_ + 1) * sizeof(int))));
        checkCuda(cudaMemcpyAsync(col_ptr_.get(), other.col_ptr_.get(), (cols_ + 1) * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream()), "memcpy");
    }
    if (nnz_ > 0) {
        row_idx_.reset(static_cast<int*>(allocate_device_memory(nnz_ * sizeof(int))));
        values_.reset(static_cast<float*>(allocate_device_memory(nnz_ * sizeof(float))));
        checkCuda(cudaMemcpyAsync(row_idx_.get(), other.row_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream()), "memcpy");
        checkCuda(cudaMemcpyAsync(values_.get(), other.values_.get(), nnz_ * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream()), "memcpy");
    }
}

SparseCSC& SparseCSC::operator=(const SparseCSC& other) {
    if (this != &other) {
        *this = SparseCSC(other);
    }
    return *this;
}

SparseCSC::SparseCSC(SparseCSC&& other) noexcept = default;
SparseCSC& SparseCSC::operator=(SparseCSC&& other) noexcept = default;

float SparseCSC::sparsity() const { return 1.0f - static_cast<float>(nnz_) / (rows_ * cols_); }

SparseCSC SparseCSC::from_dense(const Matrix& dense, float threshold) {
    std::vector<float> host_dense(dense.rows() * dense.cols());
    checkCuda(cudaMemcpy(host_dense.data(), dense.data(), host_dense.size() * sizeof(float), cudaMemcpyDeviceToHost), "from_dense memcpy");
    
    std::vector<int> col_ptr(dense.cols() + 1, 0);
    std::vector<int> row_idx;
    std::vector<float> values;
    
    for (std::size_t c = 0; c < dense.cols(); ++c) {
        for (std::size_t r = 0; r < dense.rows(); ++r) {
            float val = host_dense[r * dense.cols() + c];
            if (std::abs(val) > threshold) {
                row_idx.push_back(r);
                values.push_back(val);
            }
        }
        col_ptr[c + 1] = row_idx.size();
    }
    
    SparseCSC csc(dense.rows(), dense.cols(), values.size());
    checkCuda(cudaMemcpy(csc.col_ptr_.get(), col_ptr.data(), (dense.cols() + 1) * sizeof(int), cudaMemcpyHostToDevice), "memcpy");
    checkCuda(cudaMemcpy(csc.row_idx_.get(), row_idx.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy");
    checkCuda(cudaMemcpy(csc.values_.get(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
    
    return csc;
}

SparseCSC SparseCSC::from_csr(const SparseCSR& csr) {
    return csr_to_csc(csr);
}

Matrix SparseCSC::to_dense() const {
    Matrix dense(rows_, cols_);
    dense.fill_zero();
    std::vector<int> col_ptr(cols_ + 1);
    std::vector<int> row_idx(nnz_);
    std::vector<float> values(nnz_);
    checkCuda(cudaMemcpy(col_ptr.data(), col_ptr_.get(), (cols_ + 1) * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    checkCuda(cudaMemcpy(row_idx.data(), row_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    checkCuda(cudaMemcpy(values.data(), values_.get(), nnz_ * sizeof(float), cudaMemcpyDeviceToHost), "memcpy");
    
    std::vector<float> host_dense(rows_ * cols_, 0.0f);
    for (std::size_t c = 0; c < cols_; ++c) {
        for (int j = col_ptr[c]; j < col_ptr[c+1]; ++j) {
            host_dense[row_idx[j] * cols_ + c] = values[j];
        }
    }
    checkCuda(cudaMemcpy(dense.data(), host_dense.data(), rows_ * cols_ * sizeof(float), cudaMemcpyHostToDevice), "memcpy");
    return dense;
}

SparseCSR SparseCSC::to_csr() const {
    return csc_to_csr(*this);
}

SparseCSC SparseCSC::transpose() const {
    SparseCSC t(cols_, rows_, nnz_);
    // Essentially copying CSC to another CSC structure is like creating CSR of transposed matrix
    // Or transposing CSC gives CSR...
    return t; // Needs proper implementation, placeholder for structure
}

Matrix SparseCSC::spmv(const Matrix& x) const {
    if (x.rows() != cols_ || x.cols() != 1) throw ShapeMismatchError("spmv shape error");
    Matrix y(rows_, 1);
    y.fill_zero();
    if (cols_ > 0) {
        csc_spmv_kernel<<<detail::capped_grid(cols_, 256), 256, 0, compute_stream()>>>(
            col_ptr_.get(), row_idx_.get(), values_.get(), x.data(), y.data(), cols_);
    }
    return y;
}

Matrix SparseCSC::sparse_matmul(const Matrix& B) const {
    // Basic fallback implementation
    Matrix dense = to_dense();
    return dense * B;
}


// Conversions
SparseCSR coo_to_csr(const SparseCOO& coo) {
    SparseCSR csr(coo.rows(), coo.cols(), coo.nnz());
    int* row_counts = static_cast<int*>(allocate_device_memory((coo.rows() + 1) * sizeof(int)));
    checkCuda(cudaMemsetAsync(row_counts, 0, (coo.rows() + 1) * sizeof(int), compute_stream()), "memset");
    
    if (coo.nnz() > 0) {
        coo_to_csr_count_kernel<<<detail::capped_grid(coo.nnz(), 256), 256, 0, compute_stream()>>>(
            coo.row_indices(), row_counts, coo.nnz());
    }
    
    std::vector<int> h_counts(coo.rows() + 1, 0);
    checkCuda(cudaMemcpy(h_counts.data(), row_counts, (coo.rows() + 1) * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    
    std::vector<int> h_ptr(coo.rows() + 1, 0);
    for(size_t i = 0; i < coo.rows(); ++i) {
        h_ptr[i+1] = h_ptr[i] + h_counts[i];
    }
    
    checkCuda(cudaMemcpy(csr.row_ptr(), h_ptr.data(), (coo.rows() + 1) * sizeof(int), cudaMemcpyHostToDevice), "memcpy");
    
    // Simplification: assume COO is sorted, just copy cols and values. 
    // True scatter requires write offsets per row.
    checkCuda(cudaMemcpy(csr.col_indices(), coo.col_indices(), coo.nnz() * sizeof(int), cudaMemcpyDeviceToDevice), "memcpy");
    checkCuda(cudaMemcpy(csr.values(), coo.values(), coo.nnz() * sizeof(float), cudaMemcpyDeviceToDevice), "memcpy");
    
    free_device_memory(row_counts);
    return csr;
}

SparseCOO csr_to_coo(const SparseCSR& csr) {
    std::vector<int> h_ptr(csr.rows() + 1);
    checkCuda(cudaMemcpy(h_ptr.data(), csr.row_ptr(), (csr.rows() + 1) * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    
    std::vector<int> h_row_idx(csr.nnz());
    for(size_t r = 0; r < csr.rows(); ++r) {
        for(int j = h_ptr[r]; j < h_ptr[r+1]; ++j) {
            h_row_idx[j] = r;
        }
    }
    
    SparseCOO coo(csr.rows(), csr.cols(), csr.nnz());
    checkCuda(cudaMemcpy(const_cast<int*>(coo.row_indices()), h_row_idx.data(), csr.nnz() * sizeof(int), cudaMemcpyHostToDevice), "memcpy");
    checkCuda(cudaMemcpy(const_cast<int*>(coo.col_indices()), csr.col_indices(), csr.nnz() * sizeof(int), cudaMemcpyDeviceToDevice), "memcpy");
    checkCuda(cudaMemcpy(const_cast<float*>(coo.values()), csr.values(), csr.nnz() * sizeof(float), cudaMemcpyDeviceToDevice), "memcpy");
    return coo;
}

SparseCSC csr_to_csc(const SparseCSR& csr) {
    SparseCSC csc(csr.rows(), csr.cols(), csr.nnz());
    
    int* counts = static_cast<int*>(allocate_device_memory((csr.cols() + 1) * sizeof(int)));
    checkCuda(cudaMemsetAsync(counts, 0, (csr.cols() + 1) * sizeof(int), compute_stream()), "memset");
    
    if (csr.nnz() > 0) {
        count_nnz_per_col_kernel<<<detail::capped_grid(csr.nnz(), 256), 256, 0, compute_stream()>>>(
            csr.col_indices(), counts, csr.nnz());
    }
    
    std::vector<int> h_counts(csr.cols() + 1, 0);
    checkCuda(cudaMemcpy(h_counts.data(), counts, (csr.cols() + 1) * sizeof(int), cudaMemcpyDeviceToHost), "memcpy");
    
    std::vector<int> h_ptr(csr.cols() + 1, 0);
    for(size_t i = 0; i < csr.cols(); ++i) h_ptr[i+1] = h_ptr[i] + h_counts[i];
    
    checkCuda(cudaMemcpy(counts, h_ptr.data(), (csr.cols() + 1) * sizeof(int), cudaMemcpyHostToDevice), "memcpy"); // Now counts acts as col_ptr
    
    int* write_pos = static_cast<int*>(allocate_device_memory(csr.cols() * sizeof(int)));
    checkCuda(cudaMemsetAsync(write_pos, 0, csr.cols() * sizeof(int), compute_stream()), "memset");
    
    // Copy col_ptr to CSC
    checkCuda(cudaMemcpy(csc.col_ptr_.get(), counts, (csr.cols() + 1) * sizeof(int), cudaMemcpyDeviceToDevice), "memcpy");
    
    if (csr.rows() > 0) {
        csr_to_csc_scatter_kernel<<<detail::capped_grid(csr.rows(), 256), 256, 0, compute_stream()>>>(
            csr.row_ptr(), csr.col_indices(), csr.values(),
            csc.col_ptr_.get(), csc.row_idx_.get(), csc.values_.get(),
            write_pos, csr.rows());
    }
    
    free_device_memory(counts);
    free_device_memory(write_pos);
    return csc;
}

SparseCSR csc_to_csr(const SparseCSC& csc) {
    // Reverse operation
    return SparseCSR(csc.cols(), csc.rows(), csc.nnz()); // Placeholder
}

// SpGEMM and Addition
SparseCSR spgemm(const SparseCSR& A, const SparseCSR& B) {
    cusparseHandle_t handle = get_cusparse_handle();
    cusparseSpMatDescr_t matA, matB, matC;
    
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows(), A.cols(), A.nnz(),
                                      const_cast<int*>(A.row_ptr()), const_cast<int*>(A.col_indices()), const_cast<float*>(A.values()),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B.rows(), B.cols(), B.nnz(),
                                      const_cast<int*>(B.row_ptr()), const_cast<int*>(B.col_indices()), const_cast<float*>(B.values()),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A.rows(), B.cols(), 0,
                                      nullptr, nullptr, nullptr,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));
    
    size_t bufferSize1 = 0, bufferSize2 = 0;
    void* dBuffer1 = nullptr, *dBuffer2 = nullptr;
    
    float alpha = 1.0f, beta = 0.0f;
    
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, matA, matB, &beta, matC,
                                                  CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr));
    
    dBuffer1 = allocate_device_memory(bufferSize1);
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, matA, matB, &beta, matC,
                                                  CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));

    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, matB, &beta, matC,
                                           CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr));
    
    dBuffer2 = allocate_device_memory(bufferSize2);
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, matB, &beta, matC,
                                           CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));

    int64_t C_rows, C_cols, C_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz));
    
    SparseCSR C(A.rows(), B.cols(), C_nnz);
    
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, C.row_ptr(), C.col_indices(), C.values()));
    
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, matB, &beta, matC,
                                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
    
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    
    if (dBuffer1) free_device_memory(dBuffer1);
    if (dBuffer2) free_device_memory(dBuffer2);
    
    return C;
}

SparseCSR sparse_add(const SparseCSR& A, const SparseCSR& B, float alpha, float beta) {
    // Fallback simple addition logic or use cusparse
    return spgemm(A, B); // Simplified placeholder
}

}

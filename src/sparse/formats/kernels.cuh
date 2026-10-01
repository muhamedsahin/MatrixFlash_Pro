// Dahili uygulama parçası: src/sparse/sparse_formats.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Format çekirdekleri, cihaz deleter'ları ve stream-aware host kopyası.

namespace {

// Host-facing conversions must wait on the library's NONBLOCKING stream.
cudaError_t copy_sync(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
    if (!bytes)
        return cudaSuccess;
    auto status = cudaMemcpyAsync(dst, src, bytes, kind, compute_stream());
    if (status != cudaSuccess)
        return status;
    return cudaStreamSynchronize(compute_stream());
}

void release_int(int* p) {
    if (p)
        free_device_memory(p);
}
void release_float(float* p) {
    if (p)
        free_device_memory(p);
}

cusparseHandle_t get_cusparse_handle() {
    static thread_local cusparseHandle_t handle = nullptr;
    if (!handle) {
        cusparseCreate(&handle);
    }
    return handle;
}

#define CHECK_CUSPARSE(stat)                                                                       \
    do {                                                                                           \
        if (stat != CUSPARSE_STATUS_SUCCESS) {                                                     \
            throw matrix_pro::CudaError("cuSPARSE Error!");                                        \
        }                                                                                          \
    } while (0)

__global__ void coo_to_dense_kernel(const int* row_idx, const int* col_idx, const float* values,
                                    float* dense, std::size_t cols, std::size_t nnz) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nnz; i += blockDim.x * gridDim.x) {
        atomicAdd(&dense[row_idx[i] * cols + col_idx[i]], values[i]);
    }
}

__global__ void count_nnz_per_col_kernel(const int* col_idx, int* counts, std::size_t nnz) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nnz; i += blockDim.x * gridDim.x) {
        atomicAdd(&counts[col_idx[i]], 1);
    }
}

__global__ void csr_to_csc_scatter_kernel(const int* row_ptr, const int* col_idx,
                                          const float* values, int* csc_col_ptr, int* csc_row_idx,
                                          float* csc_values, int* write_pos, std::size_t rows) {
    auto r = blockIdx.x * blockDim.x + threadIdx.x;
    for (; r < rows; r += blockDim.x * gridDim.x) {
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
    for (; i < nnz; i += blockDim.x * gridDim.x) {
        atomicAdd(&row_counts[row_idx[i]], 1);
    }
}

__global__ void csc_spmv_kernel(const int* col_ptr, const int* row_idx, const float* values,
                                const float* x, float* y, std::size_t cols) {
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    for (; col < cols; col += blockDim.x * gridDim.x) {
        float x_col = x[col];
        for (int j = col_ptr[col]; j < col_ptr[col + 1]; ++j) {
            atomicAdd(&y[row_idx[j]], values[j] * x_col);
        }
    }
}

} // namespace

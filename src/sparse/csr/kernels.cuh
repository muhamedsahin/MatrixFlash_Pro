// Dahili uygulama parçası: src/sparse/sparse.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: CSR spmv/spmm çekirdekleri ve cihaz storage deleter'ları.

namespace {

void release_int(int* p) {
    if (p)
        free_device_memory(p);
}
void release_float(float* p) {
    if (p)
        free_device_memory(p);
}

__global__ void spmv_kernel(const int* row_ptr, const int* col_idx, const float* values,
                            const float* x, float* y, std::size_t rows) {
    const std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows)
        return;
    float sum = 0.0f;
    for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
        sum += values[j] * x[col_idx[j]];
    y[r] = sum;
}

__global__ void sparse_matmul_kernel(const int* row_ptr, const int* col_idx, const float* values,
                                     const float* b, float* c, std::size_t rows, std::size_t n) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= rows * n)
        return;
    const std::size_t r = t / n;
    const std::size_t col = t % n;
    float sum = 0.0f;
    for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j)
        sum += values[j] * b[static_cast<std::size_t>(col_idx[j]) * n + col];
    c[t] = sum;
}

} // namespace

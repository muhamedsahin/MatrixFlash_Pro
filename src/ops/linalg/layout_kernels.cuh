// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: cuSOLVER için row-major/column-major dönüşüm ve üçgen çıkarma.

namespace {

inline void checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw SolverError(std::string(operation) + ": cusolver status " +
                          std::to_string(static_cast<int>(status)));
    }
}

// Copy a row-major buffer into a column-major (Fortran) layout buffer of leading
// dimension `out_lda`. This is what cuSOLVER expects.
__global__ void row_major_to_column_major_kernel(const float* in, float* out, std::size_t rows,
                                                 std::size_t cols, std::size_t out_lda) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total)
        return;
    const auto row = index / cols;
    const auto col = index % cols;
    out[col * out_lda + row] = in[row * cols + col];
}

// Copy a column-major buffer (leading dimension `in_lda`) into row-major output.
__global__ void column_major_to_row_major_kernel(const float* in, float* out, std::size_t rows,
                                                 std::size_t cols, std::size_t in_lda) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total)
        return;
    const auto row = index / cols;
    const auto col = index % cols;
    out[row * cols + col] = in[col * in_lda + row];
}

// Extract the lower-triangular part of a column-major square matrix (lda=n) into a
// row-major square output, zero-filling the strict upper triangle.
__global__ void lower_triangular_cm_to_rm_kernel(const float* in, float* out, std::size_t n) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = n * n;
    if (index >= total)
        return;
    const auto row = index / n;
    const auto col = index % n;
    out[row * n + col] = (row >= col) ? in[col * n + row] : 0.0f;
}

// Extract the upper-triangular part (rows 0..k-1) of a column-major m x n buffer
// (lda=m) into a column-major k x n buffer (lda=k), zero-filling the lower part.
__global__ void upper_triangular_extract_kernel(const float* in, float* out, std::size_t m,
                                                std::size_t k, std::size_t n) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = k * n;
    if (index >= total)
        return;
    const auto row = index / n;
    const auto col = index % n;
    out[col * k + row] = (row <= col) ? in[col * m + row] : 0.0f;
}

} // namespace

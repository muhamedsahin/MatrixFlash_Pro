// Dahili uygulama parçası: src/ops/elementwise.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Satır/sütun vektoru broadcast; float4 hızlı yol korunur.

// float4 path when cols is a multiple of 4: one vector load of the row segment
// plus four scalar bias lookups (or one float4 bias load when col%4==0).
__global__ void add_row_vector_vec4_kernel(const float4* __restrict__ matrix,
                                           const float* __restrict__ vector,
                                           float4* __restrict__ output, std::size_t n4,
                                           std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        const std::size_t base = i * 4;
        const std::size_t col = base % cols;
        const float4 m = matrix[i];
        float4 r;
        r.x = m.x + vector[col];
        r.y = m.y + vector[col + 1];
        r.z = m.z + vector[col + 2];
        r.w = m.w + vector[col + 3];
        output[i] = r;
    }
}

__global__ void add_col_vector_kernel(const float* __restrict__ matrix,
                                      const float* __restrict__ vector, float* __restrict__ output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols)
        return;
    output[index] = matrix[index] + vector[index / cols];
}

__global__ void add_col_vector_vec4_kernel(const float4* __restrict__ matrix,
                                           const float* __restrict__ vector,
                                           float4* __restrict__ output, std::size_t n4,
                                           std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        const std::size_t row = (i * 4) / cols;
        const float v = vector[row];
        const float4 m = matrix[i];
        float4 r;
        r.x = m.x + v;
        r.y = m.y + v;
        r.z = m.z + v;
        r.w = m.w + v;
        output[i] = r;
    }
}

__global__ void mul_row_vector_kernel(const float* __restrict__ matrix,
                                      const float* __restrict__ vector, float* __restrict__ output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols)
        return;
    output[index] = matrix[index] * vector[index % cols];
}

__global__ void mul_row_vector_vec4_kernel(const float4* __restrict__ matrix,
                                           const float* __restrict__ vector,
                                           float4* __restrict__ output, std::size_t n4,
                                           std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        const std::size_t col = (i * 4) % cols;
        const float4 m = matrix[i];
        float4 r;
        r.x = m.x * vector[col];
        r.y = m.y * vector[col + 1];
        r.z = m.z * vector[col + 2];
        r.w = m.w * vector[col + 3];
        output[i] = r;
    }
}

__global__ void mul_col_vector_kernel(const float* __restrict__ matrix,
                                      const float* __restrict__ vector, float* __restrict__ output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols)
        return;
    output[index] = matrix[index] * vector[index / cols];
}

__global__ void mul_col_vector_vec4_kernel(const float4* __restrict__ matrix,
                                           const float* __restrict__ vector,
                                           float4* __restrict__ output, std::size_t n4,
                                           std::size_t cols) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        const std::size_t row = (i * 4) / cols;
        const float v = vector[row];
        const float4 m = matrix[i];
        float4 r;
        r.x = m.x * v;
        r.y = m.y * v;
        r.z = m.z * v;
        r.w = m.w * v;
        output[i] = r;
    }
}

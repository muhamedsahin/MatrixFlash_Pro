// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Merkezleme, geniş birikim, varyans ve korelasyon çekirdekleri.

__global__ void center_columns_kernel(const float* input, const float* means, float* output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows * cols)
        output[index] = input[index] - means[index % cols];
}

__global__ void sum_partial_double_kernel(const float* input, std::size_t count, double* partial) {
    __shared__ double shared[reduce_block];
    const auto tid = threadIdx.x;
    double local = 0.0;
    for (std::size_t index = blockIdx.x * reduce_block + tid; index < count;
         index += gridDim.x * reduce_block) {
        local += static_cast<double>(input[index]);
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        partial[blockIdx.x] = shared[0];
}

__global__ void centered_sq_sum_kernel(const float* input, std::size_t count, float mean,
                                       float* output) {
    __shared__ float shared[reduce_block];
    const auto tid = threadIdx.x;
    float local = 0.0f;
    for (std::size_t index = blockIdx.x * reduce_block + tid; index < count;
         index += gridDim.x * reduce_block) {
        const float diff = input[index] - mean;
        local += diff * diff;
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = shared[0];
}
__global__ void covariance_to_correlation_kernel(const float* covariance, float* correlation,
                                                 std::size_t dimensions) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= dimensions * dimensions)
        return;
    const auto row = index / dimensions;
    const auto col = index % dimensions;
    if (row == col) {
        correlation[index] = 1.0f;
        return;
    }
    const float variance_row = covariance[row * dimensions + row];
    const float variance_col = covariance[col * dimensions + col];
    correlation[index] = variance_row > 0.0f && variance_col > 0.0f
                             ? covariance[index] / sqrtf(variance_row * variance_col)
                             : 0.0f;
}

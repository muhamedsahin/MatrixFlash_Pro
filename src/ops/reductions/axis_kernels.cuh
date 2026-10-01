// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Satır ve sütun indirgemeleri; mevcut warp/shared-memory düzeni korunur.

__global__ void row_sum_kernel(const float* input, float* output, std::size_t rows,
                               std::size_t cols) {
    __shared__ float partial[reduce_block];
    const auto tid = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float sum = 0.0f;
        for (std::size_t col = tid; col < cols; col += blockDim.x)
            sum += input[row * cols + col];
        partial[tid] = sum;
        __syncthreads();
        for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0)
            output[row] = partial[0];
    }
}

__global__ void col_sum_kernel(const float* __restrict__ input, float* __restrict__ output,
                               std::size_t rows, std::size_t cols) {
    // One thread per column: consecutive threads read consecutive addresses
    // within each row, so every row pass is a coalesced transaction. Four
    // independent accumulators break the serial dependency chain so loads
    // overlap instead of stalling on latency.
    for (std::size_t col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += gridDim.x * blockDim.x) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        std::size_t row = 0;
        for (; row + 4 <= rows; row += 4) {
            const float* base = input + row * cols + col;
            s0 += base[0];
            s1 += base[cols];
            s2 += base[2 * cols];
            s3 += base[3 * cols];
        }
        for (; row < rows; ++row)
            s0 += input[row * cols + col];
        output[col] = (s0 + s1) + (s2 + s3);
    }
}

// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Trace çekirdeği ve skaler host sonucu.

__global__ void trace_kernel(const float* input, std::size_t rows, std::size_t cols,
                             float* output) {
    __shared__ float shared[reduce_block];
    const auto tid = threadIdx.x;
    float local = 0.0f;
    for (std::size_t index = tid; index < rows; index += blockDim.x) {
        const std::size_t row_index = index * cols + index;
        local += input[row_index];
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        *output = shared[0];
}

float trace(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols())
        throw InvalidArgumentError("Trace requires a square matrix");
    if (matrix.empty())
        return 0.0f;
    float* device_result = static_cast<float*>(allocate_device_memory(sizeof(float)));
    trace_kernel<<<1, reduce_block, 0, compute_stream()>>>(matrix.device_data(), matrix.rows(),
                                                           matrix.cols(), device_result);
    checkCuda(cudaGetLastError(), "trace kernel launch");
    float result = 0.0f;
    checkCuda(cudaMemcpyAsync(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "trace read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "trace synchronize");
    free_device_memory(device_result);
    return result;
}

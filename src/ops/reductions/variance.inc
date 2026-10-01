// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: İki geçişli varyans; ortalama double ile biriktirilir.

float variance(const Matrix& matrix) {
    if (matrix.empty())
        throw InvalidArgumentError("Variance is undefined for an empty matrix");
    const std::size_t count = matrix.size();
    if (count == 1)
        return 0.0f;

    // Two-pass computation for numerical stability (avoids catastrophic cancellation).
    // The mean is accumulated in double precision: a naive float sum of large
    // inputs loses low-order bits, which would poison the second pass.
    const unsigned grid = reduce_grid(count);
    double* mean_partial = static_cast<double*>(
        allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(double)));
    sum_partial_double_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(),
                                                                           count, mean_partial);
    checkCuda(cudaGetLastError(), "double sum kernel launch");

    std::vector<double> host_mean_partial(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_mean_partial.data(), mean_partial,
                              host_mean_partial.size() * sizeof(double), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "double sum read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "double sum synchronize");
    free_device_memory(mean_partial);

    double total = 0.0;
    for (double value : host_mean_partial)
        total += value;
    const float mean_value = static_cast<float>(total / static_cast<double>(count));

    float* partial =
        static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    centered_sq_sum_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), count,
                                                                        mean_value, partial);
    checkCuda(cudaGetLastError(), "centered square sum kernel launch");

    std::vector<float> host_partial(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_partial.data(), partial, host_partial.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, compute_stream()),
              "centered square sum read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "centered square sum synchronize");
    free_device_memory(partial);

    float sum_squared_diff = 0.0f;
    for (float value : host_partial)
        sum_squared_diff += value;
    return sum_squared_diff / static_cast<float>(count - 1);
}
float stddev(const Matrix& matrix) {
    return std::sqrt(variance(matrix));
}

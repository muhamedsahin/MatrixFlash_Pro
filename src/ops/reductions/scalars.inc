// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: GPU partial sonuçlarını hostta birleştirir; skaler API buradadır.

std::array<float, 5> reduce_stats(const Matrix& matrix) {
    const std::size_t count = matrix.size();
    if (count == 0) {
        return {0.0f, 0.0f, 0.0f, FLT_MAX, -FLT_MAX};
    }

    const unsigned grid = reduce_grid(count);
    float* partial = static_cast<float*>(
        allocate_device_memory(static_cast<std::size_t>(grid) * 5U * sizeof(float)));
    reduce_partial_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), count,
                                                                       partial);
    checkCuda(cudaGetLastError(), "reduce partial kernel launch");

    PinnedFloatScratch host_partial(static_cast<std::size_t>(grid) * 5U);
    checkCuda(cudaMemcpyAsync(host_partial.data(), partial,
                              static_cast<std::size_t>(grid) * 5U * sizeof(float),
                              cudaMemcpyDeviceToHost, compute_stream()),
              "reduce partial read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "reduce partial synchronize");
    free_device_memory(partial);

    const float* host = host_partial.data();

    std::array<float, 5> result = {0.0f, 0.0f, 0.0f, FLT_MAX, -FLT_MAX};
    for (std::size_t block = 0; block < static_cast<std::size_t>(grid); ++block) {
        const std::size_t offset = block * 5U;
        result[0] += host[offset + 0];
        result[1] += host[offset + 1];
        result[2] += host[offset + 2];
        result[3] = std::fmin(result[3], host[offset + 3]);
        result[4] = std::fmax(result[4], host[offset + 4]);
    }
    return result;
}

float reduce_scalar(const Matrix& matrix, int which) {
    return reduce_stats(matrix)[static_cast<std::size_t>(which)];
}

float sum(const Matrix& matrix) {
    return reduce_scalar(matrix, 0);
}
float mean(const Matrix& matrix) {
    if (matrix.empty())
        throw InvalidArgumentError("Mean is undefined for an empty matrix");
    return sum(matrix) / static_cast<float>(matrix.size());
}
float min(const Matrix& matrix) {
    return reduce_scalar(matrix, 3);
}
float max(const Matrix& matrix) {
    return reduce_scalar(matrix, 4);
}
float l1_norm(const Matrix& matrix) {
    return reduce_scalar(matrix, 2);
}
float l2_norm(const Matrix& matrix) {
    return std::sqrt(reduce_scalar(matrix, 1));
}
float frobenius_norm(const Matrix& matrix) {
    return l2_norm(matrix);
}
float abs_max(const Matrix& matrix) {
    if (matrix.empty())
        return 0.0f;
    const std::array<float, 5> stats = reduce_stats(matrix);
    return std::fmax(std::fabs(stats[3]), std::fabs(stats[4]));
}

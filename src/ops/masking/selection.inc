// Dahili uygulama parçası: src/ops/masking.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Any/all, where, apply_mask ve filtreleme; indirme noktası açıktır.

bool any(const Matrix& matrix) {
    if (matrix.empty())
        return false;
    auto* result = static_cast<unsigned int*>(allocate_device_memory(sizeof(unsigned int)));
    checkCuda(cudaMemsetAsync(result, 0, sizeof(unsigned int), compute_stream()),
              "any result reset");
    any_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), matrix.size(), result);
    checkCuda(cudaGetLastError(), "any kernel launch");
    unsigned int host_result = 0;
    checkCuda(cudaMemcpyAsync(&host_result, result, sizeof(host_result), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "any result read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "any synchronize");
    free_device_memory(result);
    return host_result != 0;
}

bool all(const Matrix& matrix) {
    if (matrix.empty())
        return true;
    auto* result = static_cast<unsigned int*>(allocate_device_memory(sizeof(unsigned int)));
    checkCuda(cudaMemsetAsync(result, 1, sizeof(unsigned int), compute_stream()),
              "all result reset");
    all_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), matrix.size(), result);
    checkCuda(cudaGetLastError(), "all kernel launch");
    unsigned int host_result = 0;
    checkCuda(cudaMemcpyAsync(&host_result, result, sizeof(host_result), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "all result read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "all synchronize");
    free_device_memory(result);
    return host_result != 0;
}

Matrix where(const Matrix& condition, const Matrix& true_value, const Matrix& false_value) {
    if (condition.rows() != true_value.rows() || condition.cols() != true_value.cols() ||
        condition.rows() != false_value.rows() || condition.cols() != false_value.cols()) {
        throw ShapeMismatchError("where() requires matching condition/branch shapes");
    }
    Matrix output(condition.rows(), condition.cols());
    where_kernel<<<static_cast<unsigned>((condition.size() + 255) / 256), 256, 0,
                   compute_stream()>>>(condition.device_data(), true_value.device_data(),
                                       false_value.device_data(), output.device_data(),
                                       condition.size());
    checkCuda(cudaGetLastError(), "where kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix where(const Matrix& condition, float true_value, float false_value) {
    Matrix output(condition.rows(), condition.cols());
    where_scalar_kernel<<<static_cast<unsigned>((condition.size() + 255) / 256), 256, 0,
                          compute_stream()>>>(condition.device_data(), true_value, false_value,
                                              output.device_data(), condition.size());
    checkCuda(cudaGetLastError(), "where scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix apply_mask(const Matrix& matrix, const Matrix& mask, float value) {
    if (matrix.rows() != mask.rows() || matrix.cols() != mask.cols()) {
        throw ShapeMismatchError("apply_mask() requires matching shapes");
    }
    Matrix output(matrix.rows(), matrix.cols());
    apply_mask_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                        compute_stream()>>>(matrix.device_data(), mask.device_data(),
                                            output.device_data(), matrix.size(), value);
    checkCuda(cudaGetLastError(), "apply_mask kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix filter_by_mask(const Matrix& matrix, const Matrix& mask) {
    if (matrix.rows() != mask.rows() || matrix.cols() != mask.cols()) {
        throw ShapeMismatchError("filter_by_mask() requires matching shapes");
    }
    const std::size_t count = matrix.size();
    auto* device_count =
        static_cast<unsigned long long*>(allocate_device_memory(sizeof(unsigned long long)));
    checkCuda(cudaMemsetAsync(device_count, 0, sizeof(unsigned long long), compute_stream()),
              "filter count reset");
    count_mask_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, compute_stream()>>>(
        mask.device_data(), count, device_count);
    checkCuda(cudaGetLastError(), "filter count kernel launch");
    unsigned long long active_count = 0;
    checkCuda(cudaMemcpyAsync(&active_count, device_count, sizeof(active_count),
                              cudaMemcpyDeviceToHost, compute_stream()),
              "filter count read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "filter count synchronize");
    free_device_memory(device_count);
    const std::size_t active = static_cast<std::size_t>(active_count);
    Matrix output(active, 1);
    if (active == 0)
        return output;

    auto* device_position =
        static_cast<unsigned long long*>(allocate_device_memory(sizeof(unsigned long long)));
    checkCuda(cudaMemsetAsync(device_position, 0, sizeof(unsigned long long), compute_stream()),
              "filter position reset");
    scatter_mask_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), mask.device_data(), output.device_data(), count, device_position);
    checkCuda(cudaGetLastError(), "filter scatter kernel launch");
    free_device_memory(device_position);
    output.mark_host_stale();
    return output;
}

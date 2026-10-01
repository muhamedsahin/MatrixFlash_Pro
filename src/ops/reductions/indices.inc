// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Argmin/argmax; eşitliklerde mevcut indeks seçimi korunur.

std::size_t argmin(const Matrix& matrix) {
    if (matrix.empty())
        throw InvalidArgumentError("argmin undefined for empty matrix");
    const std::size_t count = matrix.size();
    const unsigned grid = reduce_grid(count);

    float* partial_values =
        static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    std::size_t* partial_indices = static_cast<std::size_t*>(
        allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(std::size_t)));
    arg_partial_kernel<<<grid, reduce_block, 0, compute_stream()>>>(
        matrix.device_data(), count, 0, partial_values, partial_indices);
    checkCuda(cudaGetLastError(), "argmin partial kernel launch");

    std::vector<float> host_values(static_cast<std::size_t>(grid));
    std::vector<std::size_t> host_indices(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_values.data(), partial_values,
                              host_values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "argmin values read");
    checkCuda(cudaMemcpyAsync(host_indices.data(), partial_indices,
                              host_indices.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "argmin indices read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "argmin synchronize");
    free_device_memory(partial_values);
    free_device_memory(partial_indices);

    std::size_t best_index = host_indices.front();
    float best_value = host_values.front();
    for (std::size_t i = 1; i < host_values.size(); ++i) {
        if (host_values[i] < best_value ||
            (host_values[i] == best_value && host_indices[i] < best_index)) {
            best_index = host_indices[i];
            best_value = host_values[i];
        }
    }
    return best_index;
}

std::size_t argmax(const Matrix& matrix) {
    if (matrix.empty())
        throw InvalidArgumentError("argmax undefined for empty matrix");
    const std::size_t count = matrix.size();
    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);

    float* partial_values =
        static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    std::size_t* partial_indices = static_cast<std::size_t*>(
        allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(std::size_t)));
    arg_partial_kernel<<<grid, reduce_block, 0, compute_stream()>>>(
        matrix.device_data(), count, 1, partial_values, partial_indices);
    checkCuda(cudaGetLastError(), "argmax partial kernel launch");

    std::vector<float> host_values(static_cast<std::size_t>(grid));
    std::vector<std::size_t> host_indices(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_values.data(), partial_values,
                              host_values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "argmax values read");
    checkCuda(cudaMemcpyAsync(host_indices.data(), partial_indices,
                              host_indices.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "argmax indices read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "argmax synchronize");
    free_device_memory(partial_values);
    free_device_memory(partial_indices);

    std::size_t best_index = host_indices.front();
    float best_value = host_values.front();
    for (std::size_t i = 1; i < host_values.size(); ++i) {
        if (host_values[i] > best_value ||
            (host_values[i] == best_value && host_indices[i] < best_index)) {
            best_index = host_indices[i];
            best_value = host_values[i];
        }
    }
    return best_index;
}

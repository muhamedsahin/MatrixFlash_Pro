// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Simetrik özdeğer/özvektör çözümunun vendor yolu.

EigenResult eigen(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw SolverError("eigen() requires a square matrix");
    }
    const std::size_t n = matrix.rows();
    if (n == 0) {
        EigenResult empty;
        empty.eigenvalues = Matrix(0, 0);
        empty.eigenvectors = Matrix(0, 0);
        return empty;
    }

    float* device_a = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    float* device_w = static_cast<float*>(allocate_device_memory(n * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, n, n, n);
        checkCuda(cudaGetLastError(), "eigen conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSsyevd_bufferSize(cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_UPPER, static_cast<int>(n),
                                                  device_a, static_cast<int>(n), device_w, &lwork),
                      "eigen workspace query");
        device_work = static_cast<float*>(
            allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSsyevd(cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                                       CUBLAS_FILL_MODE_UPPER, static_cast<int>(n), device_a,
                                       static_cast<int>(n), device_w, device_work, lwork,
                                       device_info),
                      "eigen syevd");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost,
                                  compute_stream()),
                  "eigen info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "eigen info sync");
        if (info != 0) {
            throw SolverError("eigen(): eigendecomposition failed (matrix may not be symmetric)");
        }

        EigenResult result;
        result.eigenvectors = Matrix(n, n);
        column_major_to_row_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, result.eigenvectors.device_data(), n, n, n);
        checkCuda(cudaGetLastError(), "eigen eigenvectors conversion kernel");

        std::vector<float> values(n);
        checkCuda(cudaMemcpyAsync(values.data(), device_w, n * sizeof(float),
                                  cudaMemcpyDeviceToHost, compute_stream()),
                  "eigen values read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "eigen values sync");
        std::vector<float> diag(n * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i)
            diag[i * n + i] = values[i];
        result.eigenvalues = Matrix(n, n, diag);

        free_device_memory(device_a);
        free_device_memory(device_info);
        free_device_memory(device_work);
        result.eigenvectors.mark_host_stale();
        return result;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_w);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}

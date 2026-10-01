// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: SVD ve U/S/Vt sonuçlarının cihaz/host sözleşmesi.

SVDResult svd(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0) {
        SVDResult empty;
        empty.u = Matrix(m, m);
        empty.s = Matrix(0, 0);
        empty.v = Matrix(n, n);
        return empty;
    }
    const std::size_t minmn = std::min(m, n);

    float* device_a = static_cast<float*>(allocate_device_memory(m * n * sizeof(float)));
    float* device_s = static_cast<float*>(allocate_device_memory(minmn * sizeof(float)));
    float* device_u = static_cast<float*>(allocate_device_memory(m * m * sizeof(float)));
    float* device_vt = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    float* device_rwork = static_cast<float*>(allocate_device_memory(minmn * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(m * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, m, n, m);
        checkCuda(cudaGetLastError(), "svd conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSgesvd_bufferSize(cusolver_handle(), static_cast<int>(m),
                                                  static_cast<int>(n), &lwork),
                      "svd workspace query");
        device_work = static_cast<float*>(
            allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSgesvd(cusolver_handle(), 'A', 'A', static_cast<int>(m),
                                       static_cast<int>(n), device_a, static_cast<int>(m), device_s,
                                       device_u, static_cast<int>(m), device_vt,
                                       static_cast<int>(n), device_work, lwork, device_rwork,
                                       device_info),
                      "svd gesvd");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost,
                                  compute_stream()),
                  "svd info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "svd info sync");
        if (info != 0)
            throw SolverError("svd(): singular value decomposition failed to converge");

        SVDResult result;
        result.u = Matrix(m, m);
        result.v = Matrix(n, n);
        column_major_to_row_major_kernel<<<(m * m + 255) / 256, 256, 0, compute_stream()>>>(
            device_u, result.u.device_data(), m, m, m);
        checkCuda(cudaGetLastError(), "svd U conversion kernel");
        column_major_to_row_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_vt, result.v.device_data(), n, n, n);
        checkCuda(cudaGetLastError(), "svd Vt conversion kernel");
        result.v = transpose(result.v); // V = Vt^T

        std::vector<float> singular_values(minmn);
        checkCuda(cudaMemcpyAsync(singular_values.data(), device_s, minmn * sizeof(float),
                                  cudaMemcpyDeviceToHost, compute_stream()),
                  "svd singular values read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "svd singular values sync");
        std::vector<float> diag(minmn * minmn, 0.0f);
        for (std::size_t i = 0; i < minmn; ++i)
            diag[i * minmn + i] = singular_values[i];
        result.s = Matrix(minmn, minmn, diag);

        free_device_memory(device_a);
        free_device_memory(device_s);
        free_device_memory(device_u);
        free_device_memory(device_vt);
        free_device_memory(device_rwork);
        free_device_memory(device_info);
        free_device_memory(device_work);
        result.u.mark_host_stale(); // (result.v is marked by transpose above)
        return result;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_s);
        free_device_memory(device_u);
        free_device_memory(device_vt);
        free_device_memory(device_rwork);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}

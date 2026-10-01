// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: QR faktörleme; Q/R cihaz yerleşim dönüşümleri.

QRResult qr(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0) {
        QRResult empty;
        empty.q = Matrix(m, 0);
        empty.r = Matrix(0, n);
        return empty;
    }
    const std::size_t k = std::min(m, n);

    float* device_a = static_cast<float*>(allocate_device_memory(m * n * sizeof(float)));
    float* device_r = static_cast<float*>(allocate_device_memory(k * n * sizeof(float)));
    float* device_tau = static_cast<float*>(allocate_device_memory(k * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_qr_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(m * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, m, n, m);
        checkCuda(cudaGetLastError(), "qr conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSgeqrf_bufferSize(cusolver_handle(), static_cast<int>(m),
                                                  static_cast<int>(n), device_a,
                                                  static_cast<int>(m), &lwork),
                      "qr geqrf workspace query");
        device_qr_work = static_cast<float*>(
            allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSgeqrf(cusolver_handle(), static_cast<int>(m), static_cast<int>(n),
                                       device_a, static_cast<int>(m), device_tau, device_qr_work,
                                       lwork, device_info),
                      "qr geqrf");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost,
                                  compute_stream()),
                  "qr info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "qr info sync");
        if (info != 0)
            throw MatrixProError("qr(): factorization failed");

        // Preserve R before orgqr overwrites the leading columns with Q.
        upper_triangular_extract_kernel<<<(k * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, device_r, m, k, n);
        checkCuda(cudaGetLastError(), "qr R extraction kernel");

        checkCusolver(cusolverDnSorgqr_bufferSize(
                          cusolver_handle(), static_cast<int>(m), static_cast<int>(k),
                          static_cast<int>(k), device_a, static_cast<int>(m), device_tau, &lwork),
                      "qr orgqr workspace query");
        free_device_memory(device_qr_work);
        device_qr_work = static_cast<float*>(
            allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSorgqr(cusolver_handle(), static_cast<int>(m), static_cast<int>(k),
                                       static_cast<int>(k), device_a, static_cast<int>(m),
                                       device_tau, device_qr_work, lwork, device_info),
                      "qr orgqr");
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost,
                                  compute_stream()),
                  "qr orgqr info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "qr orgqr info sync");
        if (info != 0)
            throw MatrixProError("qr(): Q generation failed");

        QRResult result;
        result.q = Matrix(m, k);
        result.r = Matrix(k, n);
        column_major_to_row_major_kernel<<<(m * k + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, result.q.device_data(), m, k, m);
        checkCuda(cudaGetLastError(), "qr Q conversion kernel");
        column_major_to_row_major_kernel<<<(k * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_r, result.r.device_data(), k, n, k);
        checkCuda(cudaGetLastError(), "qr R conversion kernel");

        free_device_memory(device_a);
        free_device_memory(device_r);
        free_device_memory(device_tau);
        free_device_memory(device_info);
        free_device_memory(device_qr_work);
        result.q.mark_host_stale();
        result.r.mark_host_stale();
        return result;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_r);
        free_device_memory(device_tau);
        free_device_memory(device_info);
        free_device_memory(device_qr_work);
        throw;
    }
}

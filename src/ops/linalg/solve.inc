// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: LU faktörleme ve geri yerine koyma; geçici tamponlar hata yolunda da iade edilir.

Matrix solve(const Matrix& left, const Matrix& rhs) {
    if (left.rows() != left.cols()) {
        throw InvalidArgumentError("solve() requires a square coefficient matrix");
    }
    const std::size_t n = left.rows();
    const std::size_t k = rhs.cols();
    if (rhs.rows() != n) {
        throw ShapeMismatchError(
            "solve() requires the RHS rows to match the coefficient matrix size");
    }
    if (n == 0)
        return Matrix(0, k);

    float* device_a = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    float* device_b = static_cast<float*>(allocate_device_memory(n * k * sizeof(float)));
    int* device_ipiv = static_cast<int*>(allocate_device_memory(n * sizeof(int)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            left.device_data(), device_a, n, n, n);
        checkCuda(cudaGetLastError(), "solve A conversion kernel");
        row_major_to_column_major_kernel<<<(n * k + 255) / 256, 256, 0, compute_stream()>>>(
            rhs.device_data(), device_b, n, k, n);
        checkCuda(cudaGetLastError(), "solve b conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSgetrf_bufferSize(cusolver_handle(), static_cast<int>(n),
                                                  static_cast<int>(n), device_a,
                                                  static_cast<int>(n), &lwork),
                      "solve workspace query");
        device_work = static_cast<float*>(
            allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSgetrf(cusolver_handle(), static_cast<int>(n), static_cast<int>(n),
                                       device_a, static_cast<int>(n), device_work, device_ipiv,
                                       device_info),
                      "solve LU factorize");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost,
                                  compute_stream()),
                  "solve LU info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "solve LU info sync");
        if (info != 0) {
            throw MatrixProError("solve(): matrix is singular");
        }

        checkCusolver(cusolverDnSgetrs(cusolver_handle(), CUBLAS_OP_N, static_cast<int>(n),
                                       static_cast<int>(k), device_a, static_cast<int>(n),
                                       device_ipiv, device_b, static_cast<int>(n), device_info),
                      "solve back-substitution");
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost,
                                  compute_stream()),
                  "solve info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "solve info sync");
        if (info != 0) {
            throw MatrixProError("solve(): back-substitution failed");
        }

        Matrix output(n, k);
        column_major_to_row_major_kernel<<<(n * k + 255) / 256, 256, 0, compute_stream()>>>(
            device_b, output.device_data(), n, k, n);
        checkCuda(cudaGetLastError(), "solve output conversion kernel");

        free_device_memory(device_a);
        free_device_memory(device_b);
        free_device_memory(device_ipiv);
        free_device_memory(device_info);
        free_device_memory(device_work);
        output.mark_host_stale();
        return output;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_b);
        free_device_memory(device_ipiv);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}

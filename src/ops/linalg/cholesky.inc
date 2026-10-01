// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Pozitif tanımlı matrisin alt üçgen Cholesky faktörü.

Matrix cholesky(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw SolverError("cholesky() requires a square matrix");
    }
    const std::size_t n = matrix.rows();
    if (n == 0)
        return Matrix(0, 0);

    float* device_a = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, n, n, n);
        checkCuda(cudaGetLastError(), "cholesky conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSpotrf_bufferSize(cusolver_handle(), CUBLAS_FILL_MODE_LOWER,
                                                  static_cast<int>(n), device_a,
                                                  static_cast<int>(n), &lwork),
                      "cholesky workspace query");
        device_work = static_cast<float*>(
            allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSpotrf(cusolver_handle(), CUBLAS_FILL_MODE_LOWER,
                                       static_cast<int>(n), device_a, static_cast<int>(n),
                                       device_work, lwork, device_info),
                      "cholesky potrf");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost,
                                  compute_stream()),
                  "cholesky info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "cholesky info sync");
        if (info != 0) {
            throw SolverError("cholesky(): matrix is not positive definite");
        }

        Matrix output(n, n);
        lower_triangular_cm_to_rm_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, output.device_data(), n);
        checkCuda(cudaGetLastError(), "cholesky output extraction kernel");

        free_device_memory(device_a);
        free_device_memory(device_info);
        free_device_memory(device_work);
        output.mark_host_stale();
        return output;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}

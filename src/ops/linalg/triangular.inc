// Dahili uygulama parçası: src/ops/linalg_extended.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: LU sonuçları, triangular solve ve triangular multiply.

LUResult lu(const Matrix& A) {
    if (A.rows() != A.cols())
        throw ShapeMismatchError("Matrix must be square");
    std::size_t n = A.rows();

    Matrix LU_cm(n, n, MemoryMode::device_only);
    int num_blocks = (n * n + 255) / 256;
    detail::row_to_col_major_kernel<<<num_blocks, 256, 0, compute_stream()>>>(
        A.device_data(), LU_cm.device_data(), n, n);

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(cusolver_handle(), n, n, LU_cm.device_data(), n, &lwork);

    float* d_work;
    cudaMalloc(&d_work, lwork * sizeof(float));
    int* d_ipiv;
    cudaMalloc(&d_ipiv, n * sizeof(int));
    int* d_info;
    cudaMalloc(&d_info, sizeof(int));

    cusolverDnSgetrf(cusolver_handle(), n, n, LU_cm.device_data(), n, d_work, d_ipiv, d_info);

    LUResult result;
    result.L = Matrix(n, n, MemoryMode::device_only);
    result.U = Matrix(n, n, MemoryMode::device_only);
    result.P = Matrix(n, n, MemoryMode::device_only);

    detail::extract_L_kernel<<<num_blocks, 256, 0, compute_stream()>>>(LU_cm.device_data(),
                                                                       result.L.device_data(), n);
    detail::extract_U_kernel<<<num_blocks, 256, 0, compute_stream()>>>(LU_cm.device_data(),
                                                                       result.U.device_data(), n);
    detail::build_permutation_matrix_kernel<<<1, 1, 0, compute_stream()>>>(
        d_ipiv, result.P.device_data(), n);

    cudaFree(d_work);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    result.L.mark_host_stale();
    result.U.mark_host_stale();
    result.P.mark_host_stale();

    return result;
}

Matrix trsm(const Matrix& A, const Matrix& B, bool upper, bool left, bool unit_diag) {
    Matrix X(B.rows(), B.cols(), MemoryMode::device_only);
    cudaMemcpyAsync(X.device_data(), B.device_data(), B.rows() * B.cols() * sizeof(float),
                    cudaMemcpyDeviceToDevice, compute_stream());

    cublasFillMode_t fill = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasSideMode_t side = left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
    cublasDiagType_t diag = unit_diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    const float alpha = 1.0f;
    cublasStrsm(cublas_handle(), side, fill, CUBLAS_OP_N, diag, B.cols(), B.rows(), &alpha,
                A.device_data(), A.cols(), X.device_data(), B.cols());

    X.mark_host_stale();
    return X;
}

Matrix trmm(const Matrix& A, const Matrix& B, bool upper, bool left) {
    Matrix X(B.rows(), B.cols(), MemoryMode::device_only);
    cudaMemcpyAsync(X.device_data(), B.device_data(), B.rows() * B.cols() * sizeof(float),
                    cudaMemcpyDeviceToDevice, compute_stream());
    cublasFillMode_t fill = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasSideMode_t side = left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
    const float alpha = 1.0f;
    cublasStrmm(cublas_handle(), side, fill, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B.cols(), B.rows(),
                &alpha, A.device_data(), A.cols(), B.device_data(), B.cols(), X.device_data(),
                B.cols());
    X.mark_host_stale();
    return X;
}

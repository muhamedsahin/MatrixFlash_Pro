// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: SVD tabanlı condition number, covariance ve correlation.

float condition_number(const Matrix& matrix) {
    if (matrix.empty())
        return 1.0f;
    const auto decomposition = svd(matrix);
    const std::size_t minmn = std::min(matrix.rows(), matrix.cols());
    float max_sigma = 0.0f;
    float min_sigma = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < minmn; ++i) {
        const float sigma = decomposition.s.data()[i * minmn + i];
        max_sigma = std::max(max_sigma, sigma);
        min_sigma = std::min(min_sigma, sigma);
    }
    if (!std::isfinite(min_sigma) || min_sigma <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    return max_sigma / min_sigma;
}

Matrix covariance(const Matrix& matrix) {
    const std::size_t rows = matrix.rows();
    const std::size_t cols = matrix.cols();
    if (rows == 0 || cols == 0)
        return Matrix(rows, cols);

    Matrix means = col_sum(matrix) * (1.0f / static_cast<float>(rows));
    Matrix centered(rows, cols);
    center_columns_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                            compute_stream()>>>(matrix.device_data(), means.device_data(),
                                                centered.device_data(), rows, cols);
    checkCuda(cudaGetLastError(), "center columns kernel launch");

    Matrix result(cols, cols);
    const float alpha = 1.0f / static_cast<float>(rows > 1 ? rows - 1 : 1);
    const float beta = 0.0f;
    const auto covariance_status =
        cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(cols),
                    static_cast<int>(cols), static_cast<int>(rows), &alpha, centered.device_data(),
                    static_cast<int>(cols), centered.device_data(), static_cast<int>(cols), &beta,
                    result.device_data(), static_cast<int>(cols));
    if (covariance_status != CUBLAS_STATUS_SUCCESS)
        throw CudaError("covariance cuBLAS GEMM failed");
    result.mark_host_stale();
    return result;
}

Matrix correlation(const Matrix& matrix) {
    Matrix cov = covariance(matrix);
    const std::size_t dims = cov.rows();
    Matrix result(dims, dims);
    if (dims == 0)
        return result;

    covariance_to_correlation_kernel<<<static_cast<unsigned>((dims * dims + 255) / 256), 256, 0,
                                       compute_stream()>>>(cov.device_data(), result.device_data(),
                                                           dims);
    checkCuda(cudaGetLastError(), "correlation kernel launch");
    result.mark_host_stale();
    return result;
}

float Matrix::condition_number() const {
    return matrix_pro::condition_number(*this);
}
Matrix Matrix::covariance() const {
    return matrix_pro::covariance(*this);
}
Matrix Matrix::correlation() const {
    return matrix_pro::correlation(*this);
}

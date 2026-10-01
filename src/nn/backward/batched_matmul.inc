// Dahili uygulama parçası: src/nn/conv_backward.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Batch GEMM türevleri; aynı cuBLAS transpose/stride ayarları.

// batch_matmul backward via cuBLAS strided-batched GEMM.
// grad A = G x B^T (per batch); grad B = A^T x G (per batch).
Tensor batch_matmul_backward_left(const Tensor& grad, const Tensor& right) {
    if (grad.rank() != 3 || right.rank() != 3)
        throw InvalidArgumentError("batch_matmul backward requires rank-3 tensors");
    const auto& g = grad.shape();
    const auto& b = right.shape();
    if (g[0] != b[0] || g[2] != b[2])
        throw ShapeMismatchError("batch_matmul backward shape mismatch");
    Tensor grad_left({g[0], g[1], b[1]});
    if (grad_left.size() == 0)
        return grad_left;
    const float alpha = 1.0f, beta = 0.0f;
    const auto status = cublasSgemmStridedBatched(
        cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(b[1]), static_cast<int>(g[1]),
        static_cast<int>(b[2]), &alpha, right.device_data(), static_cast<int>(b[2]),
        static_cast<long long>(b[1] * b[2]), grad.device_data(), static_cast<int>(g[2]),
        static_cast<long long>(g[1] * g[2]), &beta, grad_left.device_data(), static_cast<int>(b[1]),
        static_cast<long long>(g[1] * b[1]), static_cast<int>(g[0]));
    if (status != CUBLAS_STATUS_SUCCESS)
        throw CudaError("cublasSgemmStridedBatched failed");
    grad_left.mark_host_stale();
    return grad_left;
}

Tensor batch_matmul_backward_right(const Tensor& grad, const Tensor& left) {
    if (grad.rank() != 3 || left.rank() != 3)
        throw InvalidArgumentError("batch_matmul backward requires rank-3 tensors");
    const auto& g = grad.shape();
    const auto& a = left.shape();
    if (g[0] != a[0] || g[1] != a[1])
        throw ShapeMismatchError("batch_matmul backward shape mismatch");
    Tensor grad_right({g[0], a[2], g[2]});
    if (grad_right.size() == 0)
        return grad_right;
    const float alpha = 1.0f, beta = 0.0f;
    const auto status = cublasSgemmStridedBatched(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(g[2]), static_cast<int>(a[2]),
        static_cast<int>(a[1]), &alpha, grad.device_data(), static_cast<int>(g[2]),
        static_cast<long long>(g[1] * g[2]), left.device_data(), static_cast<int>(a[2]),
        static_cast<long long>(a[1] * a[2]), &beta, grad_right.device_data(),
        static_cast<int>(g[2]), static_cast<long long>(a[2] * g[2]), static_cast<int>(g[0]));
    if (status != CUBLAS_STATUS_SUCCESS)
        throw CudaError("cublasSgemmStridedBatched failed");
    grad_right.mark_host_stale();
    return grad_right;
}

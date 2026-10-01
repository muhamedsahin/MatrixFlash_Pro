// Dahili uygulama parçası: src/ops/linalg_extended.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Batch solve, inverse ve determinant; mevcut yerleşimler korunur.

Tensor batch_solve(const Tensor& A, const Tensor& B) {
    // A: [batch, n, n], B: [batch, n, m]
    const auto& shape_A = A.shape();
    const auto& shape_B = B.shape();
    if (shape_A.size() != 3 || shape_B.size() != 3 || shape_A[0] != shape_B[0] ||
        shape_A[1] != shape_A[2] || shape_A[2] != shape_B[1]) {
        throw ShapeMismatchError("Invalid shapes for batch_solve");
    }

    int batch_size = shape_A[0];
    int n = shape_A[1];
    int m = shape_B[2];

    Tensor X(shape_B, MemoryMode::device_only);

    for (int i = 0; i < batch_size; ++i) {
        Matrix Ai(n, n, MemoryMode::device_only);
        cudaMemcpyAsync(Ai.device_data(), A.device_data() + i * n * n, n * n * sizeof(float),
                        cudaMemcpyDeviceToDevice, compute_stream());

        Matrix Bi(n, m, MemoryMode::device_only);
        cudaMemcpyAsync(Bi.device_data(), B.device_data() + i * n * m, n * m * sizeof(float),
                        cudaMemcpyDeviceToDevice, compute_stream());

        Matrix Xi = solve(Ai, Bi);
        cudaMemcpyAsync(X.device_data() + i * n * m, Xi.device_data(), n * m * sizeof(float),
                        cudaMemcpyDeviceToDevice, compute_stream());
    }

    X.mark_host_stale();
    return X;
}

Tensor batch_inverse(const Tensor& A) {
    const auto& shape = A.shape();
    if (shape.size() != 3 || shape[1] != shape[2])
        throw ShapeMismatchError("Invalid shape");

    int batch_size = shape[0];
    int n = shape[1];

    Tensor inv(shape, MemoryMode::device_only);
    for (int i = 0; i < batch_size; ++i) {
        Matrix Ai(n, n, MemoryMode::device_only);
        cudaMemcpyAsync(Ai.device_data(), A.device_data() + i * n * n, n * n * sizeof(float),
                        cudaMemcpyDeviceToDevice, compute_stream());

        Matrix Ai_inv = inverse(Ai);
        cudaMemcpyAsync(inv.device_data() + i * n * n, Ai_inv.device_data(), n * n * sizeof(float),
                        cudaMemcpyDeviceToDevice, compute_stream());
    }
    inv.mark_host_stale();
    return inv;
}

std::vector<float> batch_det(const Tensor& A) {
    const auto& shape = A.shape();
    if (shape.size() != 3 || shape[1] != shape[2])
        throw ShapeMismatchError("Invalid shape");

    int batch_size = shape[0];
    int n = shape[1];
    std::vector<float> dets(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        Matrix Ai(n, n, MemoryMode::device_only);
        cudaMemcpyAsync(Ai.device_data(), A.device_data() + i * n * n, n * n * sizeof(float),
                        cudaMemcpyDeviceToDevice, compute_stream());
        dets[i] = determinant(Ai);
    }

    return dets;
}

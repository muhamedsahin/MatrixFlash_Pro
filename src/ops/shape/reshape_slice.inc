// Dahili uygulama parçası: src/ops/shape.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Reshape, slice ve one-hot cihaz işlemleri.

Matrix reshape(const Matrix& matrix, std::size_t rows, std::size_t cols) {
    if (rows * cols != matrix.size())
        throw ShapeMismatchError("Reshape preserves element count");
    Matrix output(rows, cols);
    if (matrix.size() != 0) {
        // Device-to-device copy only. The host mirror is left stale on purpose:
        // copying matrix.data() here would either throw (stale input) or hand
        // the caller a mirror that no longer matches the freshly shaped device
        // buffer. download() materializes it on demand.
        checkCuda(cudaMemcpyAsync(output.device_data(), matrix.device_data(),
                                  matrix.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                                  compute_stream()),
                  "reshape device copy");
        output.mark_host_stale();
    }
    return output;
}

Matrix slice_scatter(const Matrix& grad, const Matrix& like, std::size_t row_start,
                     std::size_t row_end, std::size_t col_start, std::size_t col_end) {
    if (row_end < row_start || col_end < col_start)
        throw OutOfRangeError("Invalid slice bounds");
    if (row_end - row_start != grad.rows() || col_end - col_start != grad.cols())
        throw ShapeMismatchError("Slice scatter gradient shape mismatch");
    if (row_end > like.rows() || col_end > like.cols())
        throw OutOfRangeError("Slice scatter out of bounds");
    Matrix output = Matrix::zeros(like.rows(), like.cols());
    if (grad.size() != 0) {
        slice_scatter_kernel<<<static_cast<unsigned>((grad.size() + 255) / 256), 256, 0,
                               compute_stream()>>>(grad.device_data(), output.device_data(),
                                                   like.rows(), like.cols(), grad.rows(),
                                                   grad.cols(), row_start, col_start);
        checkCuda(cudaGetLastError(), "slice scatter kernel launch");
    }
    output.mark_host_stale();
    return output;
}

Matrix one_hot(const Matrix& indices, std::size_t classes) {
    if (classes == 0)
        throw InvalidArgumentError("one_hot requires at least one class");
    if (indices.rows() != 1 && indices.cols() != 1)
        throw InvalidArgumentError("one_hot expects a vector of indices");
    Matrix output(indices.size(), classes);
    // Only the hit position is written per row, so clear the buffer first.
    zero_device_memory(output.device_data(), output.size() * sizeof(float));
    one_hot_kernel<<<static_cast<unsigned>((indices.size() + 255) / 256), 256, 0,
                     compute_stream()>>>(indices.device_data(), output.device_data(),
                                         indices.size(), classes);
    checkCuda(cudaGetLastError(), "one hot kernel launch");
    output.mark_host_stale();
    return output;
}

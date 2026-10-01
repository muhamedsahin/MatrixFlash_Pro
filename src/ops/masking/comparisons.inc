// Dahili uygulama parçası: src/ops/masking.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Skaler/matris greater, less, equal ve not_equal.

Matrix greater(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                     compute_stream()>>>(matrix.device_data(), output.device_data(), matrix.size(),
                                         scalar, CompareOp::greater);
    checkCuda(cudaGetLastError(), "greater kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix greater(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("greater() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0,
                          compute_stream()>>>(left.device_data(), right.device_data(),
                                              output.device_data(), left.size(),
                                              CompareOp::greater);
    checkCuda(cudaGetLastError(), "greater pair kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix less(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                     compute_stream()>>>(matrix.device_data(), output.device_data(), matrix.size(),
                                         scalar, CompareOp::less);
    checkCuda(cudaGetLastError(), "less kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix less(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("less() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0,
                          compute_stream()>>>(left.device_data(), right.device_data(),
                                              output.device_data(), left.size(), CompareOp::less);
    checkCuda(cudaGetLastError(), "less pair kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix equal(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                     compute_stream()>>>(matrix.device_data(), output.device_data(), matrix.size(),
                                         scalar, CompareOp::equal);
    checkCuda(cudaGetLastError(), "equal kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix equal(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("equal() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0,
                          compute_stream()>>>(left.device_data(), right.device_data(),
                                              output.device_data(), left.size(), CompareOp::equal);
    checkCuda(cudaGetLastError(), "equal pair kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix not_equal(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                     compute_stream()>>>(matrix.device_data(), output.device_data(), matrix.size(),
                                         scalar, CompareOp::not_equal);
    checkCuda(cudaGetLastError(), "not_equal kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix not_equal(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("not_equal() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0,
                          compute_stream()>>>(left.device_data(), right.device_data(),
                                              output.device_data(), left.size(),
                                              CompareOp::not_equal);
    checkCuda(cudaGetLastError(), "not_equal pair kernel launch");
    output.mark_host_stale();
    return output;
}

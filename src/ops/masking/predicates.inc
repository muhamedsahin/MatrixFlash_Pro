// Dahili uygulama parçası: src/ops/masking.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Mantık işlemleri, isnan/isinf/is_finite.

Matrix logical_and(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("logical_and() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    logical_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(),
        LogicalOp::and_op);
    checkCuda(cudaGetLastError(), "logical_and kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix logical_or(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("logical_or() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    logical_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(),
        LogicalOp::or_op);
    checkCuda(cudaGetLastError(), "logical_or kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix logical_not(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    logical_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                     compute_stream()>>>(matrix.device_data(), matrix.device_data(),
                                         output.device_data(), matrix.size(), LogicalOp::not_op);
    checkCuda(cudaGetLastError(), "logical_not kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix isnan(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    unary_predicate_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                             compute_stream()>>>(matrix.device_data(), output.device_data(),
                                                 matrix.size(), 0);
    checkCuda(cudaGetLastError(), "isnan kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix isinf(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    unary_predicate_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                             compute_stream()>>>(matrix.device_data(), output.device_data(),
                                                 matrix.size(), 1);
    checkCuda(cudaGetLastError(), "isinf kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix is_finite(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    unary_predicate_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                             compute_stream()>>>(matrix.device_data(), output.device_data(),
                                                 matrix.size(), 2);
    checkCuda(cudaGetLastError(), "is_finite kernel launch");
    output.mark_host_stale();
    return output;
}

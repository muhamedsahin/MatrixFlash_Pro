// Dahili uygulama parçası: src/ops/shape.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Divide ve genel 2D broadcast; geri geçiş genişleyen eksenleri toplar.

Matrix divide(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols())
        throw ShapeMismatchError("Matrix shapes must match");
    Matrix output(left.rows(), left.cols());
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          output.size(), DivideOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "divide kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix divide(const Matrix& matrix, float scalar) {
    if (scalar == 0.0f)
        throw InvalidArgumentError("Division by zero scalar");
    Matrix output(matrix.rows(), matrix.cols());
    detail::launch_unary(matrix.device_data(), output.device_data(), output.size(),
                         DivideScalarOp{scalar}, compute_stream());
    checkCuda(cudaGetLastError(), "divide scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_add(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0,
                       compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), output.rows(), output.cols(),
        left.rows(), left.cols(), right.rows(), right.cols(), BroadcastOperation::add);
    checkCuda(cudaGetLastError(), "broadcast add kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_multiply(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0,
                       compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), output.rows(), output.cols(),
        left.rows(), left.cols(), right.rows(), right.cols(), BroadcastOperation::multiply);
    checkCuda(cudaGetLastError(), "broadcast multiply kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_subtract(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0,
                       compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), output.rows(), output.cols(),
        left.rows(), left.cols(), right.rows(), right.cols(), BroadcastOperation::subtract);
    checkCuda(cudaGetLastError(), "broadcast subtract kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_divide(const Matrix& left, const Matrix& right) {
    validate_broadcast(left, right);
    Matrix output(std::max(left.rows(), right.rows()), std::max(left.cols(), right.cols()));
    broadcast_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0,
                       compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), output.rows(), output.cols(),
        left.rows(), left.cols(), right.rows(), right.cols(), BroadcastOperation::divide);
    checkCuda(cudaGetLastError(), "broadcast divide kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix broadcast_backward(const Matrix& grad, const Matrix& target) {
    validate_broadcast(target, grad);
    Matrix output(target.rows(), target.cols());
    broadcast_backward_kernel<<<static_cast<unsigned>((output.size() + 255) / 256), 256, 0,
                                compute_stream()>>>(grad.device_data(), output.device_data(),
                                                    target.rows(), target.cols(), grad.rows(),
                                                    grad.cols(), target.rows(), target.cols());
    checkCuda(cudaGetLastError(), "broadcast backward kernel launch");
    output.mark_host_stale();
    return output;
}

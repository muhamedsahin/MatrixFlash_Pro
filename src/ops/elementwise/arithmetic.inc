// Dahili uygulama parçası: src/ops/elementwise.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matris/skaler ve unary fonksiyonlar; mevcut vektörize launchlar kullanilir.

Matrix add(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols(), MemoryMode::device_only);
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          left.size(), AddOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "add kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix subtract(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols(), MemoryMode::device_only);
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          left.size(), SubtractOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "subtract kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix elementwise_multiply(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw ShapeMismatchError("Matrix shapes must match");
    }
    Matrix output(left.rows(), left.cols(), MemoryMode::device_only);
    detail::launch_binary(left.device_data(), right.device_data(), output.device_data(),
                          left.size(), MultiplyOp{}, compute_stream());
    checkCuda(cudaGetLastError(), "elementwise multiply kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix multiply(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols(), MemoryMode::device_only);
    detail::launch_unary(matrix.device_data(), output.device_data(), matrix.size(),
                         MultiplyScalarOp{scalar}, compute_stream());
    checkCuda(cudaGetLastError(), "scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix add_scalar(const Matrix& matrix, float value) {
    Matrix output(matrix.rows(), matrix.cols(), MemoryMode::device_only);
    detail::launch_unary(matrix.device_data(), output.device_data(), matrix.size(),
                         AddScalarOp{value}, compute_stream());
    checkCuda(cudaGetLastError(), "add scalar kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix exp(const Matrix& matrix) {
    return unary(matrix, UnaryOp::exp);
}
Matrix log(const Matrix& matrix) {
    return unary(matrix, UnaryOp::log);
}
Matrix sqrt(const Matrix& matrix) {
    return unary(matrix, UnaryOp::sqrt);
}
Matrix abs(const Matrix& matrix) {
    return unary(matrix, UnaryOp::abs);
}
Matrix clamp(const Matrix& matrix, float low, float high) {
    return unary(matrix, UnaryOp::clamp, low, high);
}
Matrix sigmoid(const Matrix& matrix) {
    return unary(matrix, UnaryOp::sigmoid);
}
Matrix tanh(const Matrix& matrix) {
    return unary(matrix, UnaryOp::tanh);
}
Matrix pow(const Matrix& matrix, float exponent) {
    return unary(matrix, UnaryOp::pow_scalar, exponent);
}
Matrix negate(const Matrix& matrix) {
    return unary(matrix, UnaryOp::negate);
}

// Dahili uygulama parçası: src/ops/shape.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Concat/stack ve Matrix shape operatörleri.

Matrix concat(const std::vector<Matrix>& parts, std::size_t axis) {
    if (parts.empty())
        throw InvalidArgumentError("concat requires at least one matrix");
    if (axis > 1)
        throw InvalidArgumentError("concat axis must be 0 (rows) or 1 (cols)");
    for (const Matrix& part : parts) {
        if (axis == 0 && part.cols() != parts.front().cols())
            throw ShapeMismatchError("concat column mismatch");
        if (axis == 1 && part.rows() != parts.front().rows())
            throw ShapeMismatchError("concat row mismatch");
    }
    std::size_t rows = 0, cols = 0;
    for (const Matrix& part : parts) {
        rows += axis == 0 ? part.rows() : 0;
        cols += axis == 1 ? part.cols() : 0;
    }
    rows = axis == 0 ? rows : parts.front().rows();
    cols = axis == 1 ? cols : parts.front().cols();
    Matrix output = Matrix::zeros(rows, cols);
    std::size_t offset = 0;
    for (const Matrix& part : parts) {
        if (part.size() == 0)
            continue;
        // Both axes copy device-to-device only and leave the host mirror
        // stale; this keeps axis 0 and axis 1 behaviour identical and works
        // with parts whose host mirror is stale (e.g. outputs of GPU ops).
        if (axis == 0) {
            checkCuda(cudaMemcpyAsync(output.device_data() + offset * cols, part.device_data(),
                                      part.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                                      compute_stream()),
                      "concat row copy");
            offset += part.rows();
        } else {
            copy_strided_kernel<<<static_cast<unsigned>((part.size() + 255) / 256), 256, 0,
                                  compute_stream()>>>(part.device_data(), output.device_data(),
                                                      part.rows(), part.cols(), part.cols(), cols,
                                                      offset);
            checkCuda(cudaGetLastError(), "concat column copy");
            offset += part.cols();
        }
    }
    output.mark_host_stale();
    return output;
}

Matrix stack(const std::vector<Matrix>& parts) {
    if (parts.empty())
        throw InvalidArgumentError("stack requires at least one matrix");
    std::vector<Matrix> flattened;
    flattened.reserve(parts.size());
    for (const Matrix& part : parts)
        flattened.push_back(reshape(part, 1, part.size()));
    return concat(flattened, 0);
}

Matrix Matrix::operator/(const Matrix& other) const {
    return divide(*this, other);
}
Matrix Matrix::operator/(float scalar) const {
    return divide(*this, scalar);
}
Matrix Matrix::reshape(std::size_t rows, std::size_t cols) const {
    return matrix_pro::reshape(*this, rows, cols);
}

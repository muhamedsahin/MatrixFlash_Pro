// Dahili uygulama parçası: src/ops/elementwise.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Satır/sütun ekleme ve çarpma; aynı kernel ve launch geometrisi.

Matrix add_row_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.cols())
        throw ShapeMismatchError("Row vector length must equal column count");
    Matrix output(matrix.rows(), matrix.cols(), MemoryMode::device_only);
    if ((matrix.cols() % 4) == 0 && matrix.size() >= 4 &&
        detail::is_aligned4(matrix.device_data()) && detail::is_aligned4(output.device_data())) {
        const std::size_t n4 = matrix.size() / 4;
        add_row_vector_vec4_kernel<<<detail::capped_grid(n4, 256), 256, 0, compute_stream()>>>(
            reinterpret_cast<const float4*>(matrix.device_data()), vector.device_data(),
            reinterpret_cast<float4*>(output.device_data()), n4, matrix.cols());
    } else {
        add_row_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                                compute_stream()>>>(matrix.device_data(), vector.device_data(),
                                                    output.device_data(), matrix.rows(),
                                                    matrix.cols());
    }
    checkCuda(cudaGetLastError(), "add row vector kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix add_col_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.rows())
        throw ShapeMismatchError("Column vector length must equal row count");
    Matrix output(matrix.rows(), matrix.cols(), MemoryMode::device_only);
    if ((matrix.cols() % 4) == 0 && matrix.size() >= 4 &&
        detail::is_aligned4(matrix.device_data()) && detail::is_aligned4(output.device_data())) {
        const std::size_t n4 = matrix.size() / 4;
        add_col_vector_vec4_kernel<<<detail::capped_grid(n4, 256), 256, 0, compute_stream()>>>(
            reinterpret_cast<const float4*>(matrix.device_data()), vector.device_data(),
            reinterpret_cast<float4*>(output.device_data()), n4, matrix.cols());
    } else {
        add_col_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                                compute_stream()>>>(matrix.device_data(), vector.device_data(),
                                                    output.device_data(), matrix.rows(),
                                                    matrix.cols());
    }
    checkCuda(cudaGetLastError(), "add col vector kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix multiply_row_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.cols())
        throw ShapeMismatchError("Row vector length must equal column count");
    Matrix output(matrix.rows(), matrix.cols(), MemoryMode::device_only);
    if ((matrix.cols() % 4) == 0 && matrix.size() >= 4 &&
        detail::is_aligned4(matrix.device_data()) && detail::is_aligned4(output.device_data())) {
        const std::size_t n4 = matrix.size() / 4;
        mul_row_vector_vec4_kernel<<<detail::capped_grid(n4, 256), 256, 0, compute_stream()>>>(
            reinterpret_cast<const float4*>(matrix.device_data()), vector.device_data(),
            reinterpret_cast<float4*>(output.device_data()), n4, matrix.cols());
    } else {
        mul_row_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                                compute_stream()>>>(matrix.device_data(), vector.device_data(),
                                                    output.device_data(), matrix.rows(),
                                                    matrix.cols());
    }
    checkCuda(cudaGetLastError(), "multiply row vector kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix multiply_col_vector(const Matrix& matrix, const Matrix& vector) {
    if (vector.size() != matrix.rows())
        throw ShapeMismatchError("Column vector length must equal row count");
    Matrix output(matrix.rows(), matrix.cols(), MemoryMode::device_only);
    if ((matrix.cols() % 4) == 0 && matrix.size() >= 4 &&
        detail::is_aligned4(matrix.device_data()) && detail::is_aligned4(output.device_data())) {
        const std::size_t n4 = matrix.size() / 4;
        mul_col_vector_vec4_kernel<<<detail::capped_grid(n4, 256), 256, 0, compute_stream()>>>(
            reinterpret_cast<const float4*>(matrix.device_data()), vector.device_data(),
            reinterpret_cast<float4*>(output.device_data()), n4, matrix.cols());
    } else {
        mul_col_vector_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0,
                                compute_stream()>>>(matrix.device_data(), vector.device_data(),
                                                    output.device_data(), matrix.rows(),
                                                    matrix.cols());
    }
    checkCuda(cudaGetLastError(), "multiply col vector kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix operator*(float scalar, const Matrix& matrix) {
    return matrix * scalar;
}

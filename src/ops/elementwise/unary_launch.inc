// Dahili uygulama parçası: src/ops/elementwise.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Unary işlemler için tek cihaz launch girişi.

Matrix unary(const Matrix& matrix, UnaryOp op, float a = 0.0f, float b = 0.0f) {
    Matrix output(matrix.rows(), matrix.cols(), MemoryMode::device_only);
    detail::launch_unary(matrix.device_data(), output.device_data(), matrix.size(),
                         UnaryDispatcher{op, a, b}, compute_stream());
    checkCuda(cudaGetLastError(), "unary kernel launch");
    output.mark_host_stale();
    return output;
}

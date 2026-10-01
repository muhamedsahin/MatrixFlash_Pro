// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Satır/sütun toplamlarını GPU matrisi olarak döndürür.

Matrix row_sum(const Matrix& matrix) {
    Matrix output(matrix.rows(), 1);
    // Cap the grid (SM count * 4) and loop rows inside the kernel so very tall
    // matrices don't launch absurd block counts.
    unsigned grid = matrix.rows() == 0 ? 1 : static_cast<unsigned>(matrix.rows());
    const unsigned max_blocks = static_cast<unsigned>(matrix_pro::sm_count()) * 4u;
    if (grid > max_blocks)
        grid = max_blocks;
    row_sum_kernel<<<grid, reduce_block, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "row sum kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix col_sum(const Matrix& matrix) {
    Matrix output(1, matrix.cols());
    const unsigned grid =
        matrix.cols() == 0 ? 1 : static_cast<unsigned>((matrix.cols() + 255) / 256);
    col_sum_kernel<<<grid, 256, 0, compute_stream()>>>(matrix.device_data(), output.device_data(),
                                                       matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "col sum kernel launch");
    output.mark_host_stale();
    return output;
}

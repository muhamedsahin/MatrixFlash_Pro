// Dahili uygulama parçası: src/sparse/sparse.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Spmv, sparse-dense multiply ve sparse outer accumulation.

Matrix spmv(const SparseCSR& a, const Matrix& x) {
    if (x.cols() != 1 || x.rows() != a.cols())
        throw ShapeMismatchError("spmv requires (m x n) sparse and (n x 1) dense");
    Matrix y(a.rows(), 1);
    if (a.empty()) {
        y.fill(0);
        return y;
    }
    spmv_kernel<<<(a.rows() + 255) / 256, 256, 0, compute_stream()>>>(
        a.row_ptr(), a.col_idx(), a.values(), x.device_data(), y.device_data(), a.rows());
    checkCuda(cudaGetLastError(), "spmv kernel launch");
    y.mark_host_stale();
    return y;
}

Matrix sparse_matmul(const SparseCSR& a, const Matrix& b) {
    if (b.rows() != a.cols())
        throw ShapeMismatchError("sparse_matmul inner dimensions must match");
    Matrix c(a.rows(), b.cols());
    if (a.empty() || c.empty()) {
        c.fill(0);
        return c;
    }
    sparse_matmul_kernel<<<(c.size() + 255) / 256, 256, 0, compute_stream()>>>(
        a.row_ptr(), a.col_idx(), a.values(), b.device_data(), c.device_data(), a.rows(), b.cols());
    checkCuda(cudaGetLastError(), "sparse_matmul kernel launch");
    c.mark_host_stale();
    return c;
}

void sparse_outer_accumulate(SparseCSR& acc, const Matrix& a, const Matrix& b) {
    if (a.rows() != acc.rows() || b.cols() != acc.cols())
        throw ShapeMismatchError("sparse_outer_accumulate shape mismatch");
    if (a.cols() != b.rows())
        throw ShapeMismatchError("sparse_outer_accumulate inner dimensions must match");
    if (acc.empty())
        return;

    Matrix product = a * b;
    product.download();

    const std::vector<int> row_ptr = acc.host_row_ptr();
    const std::vector<int> col_idx = acc.host_col_idx();
    std::vector<float> values = acc.host_values();
    for (std::size_t row = 0; row + 1 < row_ptr.size(); ++row) {
        for (int index = row_ptr[row]; index < row_ptr[row + 1]; ++index) {
            const std::size_t col =
                static_cast<std::size_t>(col_idx[static_cast<std::size_t>(index)]);
            values[static_cast<std::size_t>(index)] += product.at(row, col);
        }
    }

    checkCuda(cudaMemcpy(acc.values(), values.data(), values.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "sparse_outer_accumulate value upload");
    acc.synchronize();
}

// Dahili uygulama parçası: src/sparse/sparse_formats.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: alpha*A+beta*B birleştirme ve mevcut duplicate davranışı.

SparseCSR sparse_add(const SparseCSR& A, const SparseCSR& B, float alpha, float beta) {
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw ShapeMismatchError("sparse_add shape mismatch");
    std::vector<int> rows, cols;
    std::vector<float> values;
    for (auto item : {std::make_pair(&A, alpha), std::make_pair(&B, beta)}) {
        auto ptr = item.first->host_row_ptr(), idx = item.first->host_col_idx();
        auto v = item.first->host_values();
        for (std::size_t r = 0; r < A.rows(); ++r)
            for (int j = ptr[r]; j < ptr[r + 1]; ++j) {
                rows.push_back(static_cast<int>(r));
                cols.push_back(idx[j]);
                values.push_back(item.second * v[j]);
            }
    }
    return SparseCSR::from_coo(A.rows(), A.cols(), rows, cols, values);
}

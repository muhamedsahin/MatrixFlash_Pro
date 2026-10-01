// Dahili uygulama parçası: src/ops/gemm/gemm_plan.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: gemm/gemm_into; cihaz çıkış aynasını bayat olarak işaretler.

void gemm_into(const Matrix& left, const Matrix& right, Matrix& output, GemmOptions o, float alpha,
               float beta, const Matrix* bias) {
    const auto m = o.transpose_left ? left.cols() : left.rows(),
               k = o.transpose_left ? left.rows() : left.cols();
    const auto n = o.transpose_right ? right.rows() : right.cols();
    if (k != (o.transpose_right ? right.cols() : right.rows()))
        throw ShapeMismatchError("GEMM inner dimension mismatch");
    cached(m, n, k, o, current_device(), compute_stream())
        .execute(left, right, output, alpha, beta, bias);
}
Matrix gemm(const Matrix& left, const Matrix& right, GemmOptions o, float alpha,
            const Matrix* bias) {
    const auto m = o.transpose_left ? left.cols() : left.rows(),
               n = o.transpose_right ? right.rows() : right.cols();
    checked_dim(m);
    checked_dim(n);
    if ((o.transpose_left ? left.rows() : left.cols()) !=
        (o.transpose_right ? right.cols() : right.rows()))
        throw ShapeMismatchError("GEMM inner dimension mismatch");
    Matrix output(m, n, MemoryMode::device_only);
    gemm_into(left, right, output, o, alpha, 0, bias);
    return output;
}

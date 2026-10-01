// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matrix üzerindeki çözüm metotlarının API yönlendirmesi.

Matrix Matrix::solve(const Matrix& rhs) const {
    return matrix_pro::solve(*this, rhs);
}
QRResult Matrix::qr() const {
    return matrix_pro::qr(*this);
}
SVDResult Matrix::svd() const {
    return matrix_pro::svd(*this);
}
Matrix Matrix::cholesky() const {
    return matrix_pro::cholesky(*this);
}
EigenResult Matrix::eigen() const {
    return matrix_pro::eigen(*this);
}
Matrix Matrix::pinv() const {
    return matrix_pro::pinv(*this);
}
std::size_t Matrix::rank() const {
    return matrix_pro::rank(*this);
}
Matrix Matrix::solve_least_squares(const Matrix& rhs) const {
    return matrix_pro::solve_least_squares(*this, rhs);
}

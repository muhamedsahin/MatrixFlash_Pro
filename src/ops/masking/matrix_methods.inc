// Dahili uygulama parçası: src/ops/masking.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matrix mask metotlarının fonksiyon API yönlendirmesi.

Matrix Matrix::greater(float value) const {
    return matrix_pro::greater(*this, value);
}
Matrix Matrix::greater(const Matrix& other) const {
    return matrix_pro::greater(*this, other);
}
Matrix Matrix::less(float value) const {
    return matrix_pro::less(*this, value);
}
Matrix Matrix::less(const Matrix& other) const {
    return matrix_pro::less(*this, other);
}
Matrix Matrix::equal(float value) const {
    return matrix_pro::equal(*this, value);
}
Matrix Matrix::equal(const Matrix& other) const {
    return matrix_pro::equal(*this, other);
}
Matrix Matrix::not_equal(float value) const {
    return matrix_pro::not_equal(*this, value);
}
Matrix Matrix::not_equal(const Matrix& other) const {
    return matrix_pro::not_equal(*this, other);
}
Matrix Matrix::logical_and(const Matrix& other) const {
    return matrix_pro::logical_and(*this, other);
}
Matrix Matrix::logical_or(const Matrix& other) const {
    return matrix_pro::logical_or(*this, other);
}
Matrix Matrix::logical_not() const {
    return matrix_pro::logical_not(*this);
}
Matrix Matrix::isnan() const {
    return matrix_pro::isnan(*this);
}
Matrix Matrix::isinf() const {
    return matrix_pro::isinf(*this);
}
Matrix Matrix::is_finite() const {
    return matrix_pro::is_finite(*this);
}
bool Matrix::any() const {
    return matrix_pro::any(*this);
}
bool Matrix::all() const {
    return matrix_pro::all(*this);
}
Matrix Matrix::apply_mask(const Matrix& mask, float value) const {
    return matrix_pro::apply_mask(*this, mask, value);
}
Matrix Matrix::filter_by_mask(const Matrix& mask) const {
    return matrix_pro::filter_by_mask(*this, mask);
}
Matrix Matrix::where(const Matrix& condition, const Matrix& true_value, const Matrix& false_value) {
    return matrix_pro::where(condition, true_value, false_value);
}
Matrix Matrix::where(const Matrix& condition, float true_value, float false_value) {
    return matrix_pro::where(condition, true_value, false_value);
}

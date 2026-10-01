// Dahili uygulama parçası: src/ops/elementwise.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matrix operator ve metot sarmalayıcılari.

// --- Matrix method wrappers ---
Matrix Matrix::operator+() const {
    return *this;
}
Matrix Matrix::operator-() const {
    return matrix_pro::negate(*this);
}
Matrix Matrix::operator+(const Matrix& other) const {
    return matrix_pro::add(*this, other);
}
Matrix Matrix::operator-(const Matrix& other) const {
    return matrix_pro::subtract(*this, other);
}
Matrix Matrix::operator*(float scalar) const {
    return matrix_pro::multiply(*this, scalar);
}
Matrix Matrix::elementwise_multiply(const Matrix& other) const {
    return matrix_pro::elementwise_multiply(*this, other);
}
Matrix Matrix::add_scalar(float value) const {
    return matrix_pro::add_scalar(*this, value);
}

Matrix Matrix::exp() const {
    return matrix_pro::exp(*this);
}
Matrix Matrix::log() const {
    return matrix_pro::log(*this);
}
Matrix Matrix::sqrt() const {
    return matrix_pro::sqrt(*this);
}
Matrix Matrix::abs() const {
    return matrix_pro::abs(*this);
}
Matrix Matrix::clamp(float low, float high) const {
    return matrix_pro::clamp(*this, low, high);
}
Matrix Matrix::sigmoid() const {
    return matrix_pro::sigmoid(*this);
}
Matrix Matrix::tanh() const {
    return matrix_pro::tanh(*this);
}
Matrix Matrix::pow(float exponent) const {
    return matrix_pro::pow(*this, exponent);
}

Matrix Matrix::add_row_vector(const Matrix& v) const {
    return matrix_pro::add_row_vector(*this, v);
}
Matrix Matrix::add_col_vector(const Matrix& v) const {
    return matrix_pro::add_col_vector(*this, v);
}
Matrix Matrix::multiply_row_vector(const Matrix& v) const {
    return matrix_pro::multiply_row_vector(*this, v);
}
Matrix Matrix::multiply_col_vector(const Matrix& v) const {
    return matrix_pro::multiply_col_vector(*this, v);
}

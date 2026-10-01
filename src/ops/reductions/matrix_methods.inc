// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matrix metotlarını serbest fonksiyon API ile eşler.

float Matrix::sum() const {
    return matrix_pro::sum(*this);
}
float Matrix::mean() const {
    return matrix_pro::mean(*this);
}
float Matrix::min() const {
    return matrix_pro::min(*this);
}
float Matrix::max() const {
    return matrix_pro::max(*this);
}
std::size_t Matrix::argmin() const {
    return matrix_pro::argmin(*this);
}
std::size_t Matrix::argmax() const {
    return matrix_pro::argmax(*this);
}
float Matrix::variance() const {
    return matrix_pro::variance(*this);
}
float Matrix::stddev() const {
    return matrix_pro::stddev(*this);
}
float Matrix::l1_norm() const {
    return matrix_pro::l1_norm(*this);
}
float Matrix::l2_norm() const {
    return matrix_pro::l2_norm(*this);
}
float Matrix::frobenius_norm() const {
    return matrix_pro::frobenius_norm(*this);
}
float Matrix::abs_max() const {
    return matrix_pro::abs_max(*this);
}
float Matrix::trace() const {
    return matrix_pro::trace(*this);
}
Matrix Matrix::row_sum() const {
    return matrix_pro::row_sum(*this);
}
Matrix Matrix::col_sum() const {
    return matrix_pro::col_sum(*this);
}

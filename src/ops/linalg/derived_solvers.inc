// Dahili uygulama parçası: src/ops/linalg.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Pseudoinverse, sayısal rank ve least-squares bileşimi.

Matrix pinv(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0)
        return Matrix(n, m);

    const SVDResult decomposition = svd(matrix);
    const std::size_t minmn = std::min(m, n);

    // Tolerance mirrors LAPACK's rcond-based truncation rule.
    float max_sigma = 0.0f;
    for (std::size_t i = 0; i < minmn; ++i) {
        max_sigma = std::max(max_sigma, decomposition.s.data()[i * minmn + i]);
    }
    const float tolerance =
        static_cast<float>(std::max(m, n)) * std::numeric_limits<float>::epsilon() * max_sigma;

    std::vector<float> pseudo(n * m, 0.0f); // n x m diagonal-ish inverse singular matrix
    for (std::size_t i = 0; i < minmn; ++i) {
        const float sigma = decomposition.s.data()[i * minmn + i];
        if (sigma > tolerance)
            pseudo[i * m + i] = 1.0f / sigma;
    }
    Matrix pseudo_matrix(n, m, pseudo);

    // A^+ = V * S^+ * U^T
    Matrix ut = transpose(decomposition.u);
    return multiply(multiply(decomposition.v, pseudo_matrix), ut);
}

std::size_t rank(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0)
        return 0;

    const SVDResult decomposition = svd(matrix);
    const std::size_t minmn = std::min(m, n);

    float max_sigma = 0.0f;
    for (std::size_t i = 0; i < minmn; ++i) {
        max_sigma = std::max(max_sigma, decomposition.s.data()[i * minmn + i]);
    }
    const float tolerance =
        static_cast<float>(std::max(m, n)) * std::numeric_limits<float>::epsilon() * max_sigma;

    std::size_t count = 0;
    for (std::size_t i = 0; i < minmn; ++i) {
        if (decomposition.s.data()[i * minmn + i] > tolerance)
            ++count;
    }
    return count;
}

Matrix solve_least_squares(const Matrix& left, const Matrix& rhs) {
    if (rhs.rows() != left.rows()) {
        throw ShapeMismatchError("solve_least_squares() requires matching rows between A and b");
    }
    if (left.rows() == 0 || left.cols() == 0)
        return Matrix(left.cols(), rhs.cols());
    return multiply(pinv(left), rhs);
}

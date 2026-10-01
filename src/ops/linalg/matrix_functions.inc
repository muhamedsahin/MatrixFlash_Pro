// Dahili uygulama parçası: src/ops/linalg_extended.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matrix power, exponential ve log-determinant.

Matrix matrix_power(const Matrix& A, int exponent) {
    if (A.rows() != A.cols())
        throw ShapeMismatchError("matrix_power requires square input");
    std::uint64_t power = exponent < 0
                              ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(exponent))
                              : static_cast<std::uint64_t>(exponent);
    Matrix result = Matrix::identity(A.rows());
    if (!power)
        return result;
    Matrix base = exponent < 0 ? inverse(A) : Matrix(A);
    while (power) {
        if (power & 1)
            result = gemm(result, base);
        power >>= 1;
        if (power)
            base = gemm(base, base);
    }
    return result;
}

Matrix matrix_exp(const Matrix& A, int order) {
    if (A.rows() != A.cols())
        throw ShapeMismatchError("matrix_exp requires square input");
    if (order < 1 || order > 64)
        throw InvalidArgumentError("matrix_exp order must be in [1,64]");
    if (A.empty())
        return Matrix(0, 0, MemoryMode::device_only);
    // Infinity norm bounds the spectral radius and governs scaling. FP32 GEMM
    // forbids implicit TF32 input conversion throughout this scientific path.
    const float norm = matrix_pro::abs(A).row_sum().max();
    if (!std::isfinite(norm))
        throw InvalidArgumentError("matrix_exp requires finite input and norm");
    const double threshold = order == 13 ? 4.25 : 0.5;
    const int squarings =
        norm > threshold ? static_cast<int>(std::ceil(std::log2(norm / threshold))) : 0;
    Matrix a = A * std::ldexp(1.0f, -squarings), identity = Matrix::identity(A.rows()), result;
    if (order == 13) {
        // [13/13] Pade; coefficients normalized by b0 keep FP32 intermediates
        // away from the huge unnormalized coefficients (~6.5e16).
        const double coefficients[] = {64764752532480000.,
                                       32382376266240000.,
                                       7771770303897600.,
                                       1187353796428800.,
                                       129060195264000.,
                                       10559470521600.,
                                       670442572800.,
                                       33522128640.,
                                       1323241920.,
                                       40840800.,
                                       960960.,
                                       16380.,
                                       182.,
                                       1.};
        float b[14];
        for (int i = 0; i < 14; ++i)
            b[i] = static_cast<float>(coefficients[i] / coefficients[0]);
        Matrix a2 = gemm(a, a), a4 = gemm(a2, a2), a6 = gemm(a4, a2);
        Matrix u = gemm(a, gemm(a6, a6 * b[13] + a4 * b[11] + a2 * b[9]) + a6 * b[7] + a4 * b[5] +
                               a2 * b[3] + identity * b[1]);
        Matrix v = gemm(a6, a6 * b[12] + a4 * b[10] + a2 * b[8]) + a6 * b[6] + a4 * b[4] +
                   a2 * b[2] + identity;
        LUFactorization denominator(v - u, A.cols());
        result = denominator.solve(v + u);
    } else {
        result = identity;
        Matrix term = identity;
        for (int i = 1; i <= order; ++i) {
            term = gemm(term, a) * (1.0f / i);
            result = result + term;
        }
    }
    for (int i = 0; i < squarings; ++i)
        result = gemm(result, result);
    result.mark_host_stale();
    return result;
}

float log_determinant(const Matrix& A) {
    return static_cast<float>(slogdet(A).log_abs_det);
}

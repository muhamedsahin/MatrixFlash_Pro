#pragma once
#include "matrix_pro/core/matrix.hpp"
#include <memory>

namespace matrix_pro {
struct SLogDetResult {
    int sign = 1;
    double log_abs_det = 0;
};

// Factors A once, reuses LU/pivots and column-major RHS scratch for subsequent
// solves. Bound to its creating device/thread/stream; no concurrent execution.
// solve_into has no allocations and accepts up to max_rhs columns.
// Construction synchronizes for singularity checking and signed log determinant.
class LUFactorization {
public:
    explicit LUFactorization(const Matrix& coefficients, std::size_t max_rhs = 1);
    ~LUFactorization();
    LUFactorization(LUFactorization&&) noexcept;
    LUFactorization& operator=(LUFactorization&&) noexcept;
    LUFactorization(const LUFactorization&) = delete;
    LUFactorization& operator=(const LUFactorization&) = delete;
    Matrix solve(const Matrix& rhs, bool transpose = false);
    void solve_into(const Matrix& rhs, Matrix& output, bool transpose = false);
    SLogDetResult slogdet() const;
    bool singular() const noexcept;
    std::size_t size() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Unlike determinant(), this does not overflow/underflow when the product of
// diagonal entries is outside FP32 range. Singular -> {0, -infinity}.
// The determinant of a 0x0 matrix is defined as 1 -> {1, 0}.
SLogDetResult slogdet(const Matrix& coefficients);
} // namespace matrix_pro

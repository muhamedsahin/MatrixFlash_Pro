#pragma once

// Advanced linear algebra (cuSOLVER-accelerated): decompositions, solvers and
// rank-deficient least squares. All results are returned as float32 Matrix
// values; the factor shapes follow the thin LAPACK conventions documented per
// struct below.

#include <cstddef>

#include "matrix_pro/core/matrix.hpp"

namespace matrix_pro {

// --- decomposition result containers ---
struct QRResult {
    Matrix q;   // orthogonal factor (m x min(m,n))
    Matrix r;   // upper triangular factor (min(m,n) x n)
};

struct SVDResult {
    Matrix u;   // left singular vectors (m x m)
    Matrix s;   // diagonal matrix of singular values (min(m,n) x min(m,n))
    Matrix v;   // right singular vectors (n x n)
};

struct EigenResult {
    Matrix eigenvalues;   // diagonal matrix of eigenvalues (n x n), ascending
    Matrix eigenvectors;  // orthonormal eigenvectors, one per column (n x n)
};

// --- solvers ---
// Solves A*x = b directly (getrf + getrs). A must be square; b may be a vector
// (n x 1) or a set of n right-hand sides (n x k). Returns a matrix shaped like b.
Matrix solve(const Matrix& left, const Matrix& rhs);
// Least-squares solution of A*x = b for possibly rank-deficient /
// rectangular systems, via the SVD pseudo-inverse.
Matrix solve_least_squares(const Matrix& left, const Matrix& rhs);

// --- decompositions ---
// Thin QR decomposition: A = Q * R.
QRResult qr(const Matrix& matrix);
// Singular value decomposition: A = U * S * V^T.
SVDResult svd(const Matrix& matrix);
// Cholesky decomposition A = L * L^T for a symmetric positive definite matrix.
// Returns the lower-triangular factor L.
Matrix cholesky(const Matrix& matrix);
// Symmetric eigendecomposition A = V * D * V^T (A must be symmetric).
EigenResult eigen(const Matrix& matrix);
// Moore-Penrose pseudo-inverse, derived through the SVD.
Matrix pinv(const Matrix& matrix);
// Numeric rank of the matrix, derived through the SVD.
std::size_t rank(const Matrix& matrix);

} // namespace matrix_pro
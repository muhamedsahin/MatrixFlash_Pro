#pragma once
#include <cstddef>
#include <vector>

namespace matrix_pro {

class Matrix;
class Tensor;

struct LUResult {
    Matrix L;
    Matrix U;
    Matrix P;  // permutation matrix
};

// LU decomposition: PA = LU
LUResult lu(const Matrix& A);

// Triangular solve: solve AX=B where A is triangular
// upper=true for upper triangular, left=true for AX=B (vs XA=B)
Matrix trsm(const Matrix& A, const Matrix& B, bool upper = true, bool left = true, bool unit_diag = false);

// Triangular matrix-matrix multiply
Matrix trmm(const Matrix& A, const Matrix& B, bool upper = true, bool left = true);

// Matrix power via repeated squaring: A^n
Matrix matrix_power(const Matrix& A, int n);

// Matrix exponential via Padé approximation
Matrix matrix_exp(const Matrix& A, int order = 13);

// Log determinant (numerically stable via LU)
float log_determinant(const Matrix& A);

// Batch linear algebra on rank-3 tensors (batch of square matrices)
Tensor batch_solve(const Tensor& A, const Tensor& B);
Tensor batch_inverse(const Tensor& A);
std::vector<float> batch_det(const Tensor& A);

}

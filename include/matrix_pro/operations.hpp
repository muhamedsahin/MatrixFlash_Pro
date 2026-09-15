#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "tensor.hpp"

namespace matrix_pro {

// --- advanced linear algebra result containers ---
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

// --- basic elementwise ---
Matrix add(const Matrix& left, const Matrix& right);
Matrix subtract(const Matrix& left, const Matrix& right);
Matrix elementwise_multiply(const Matrix& left, const Matrix& right);
Matrix multiply(const Matrix& left, const Matrix& right);   // matmul
Matrix matmul_half(const Matrix& left, const Matrix& right); // FP16 inputs, FP32 output
Matrix matmul_double_accumulate(const Matrix& left, const Matrix& right); // FP64 accumulation, FP32 output
Matrix multiply(const Matrix& matrix, float scalar);
Matrix add_scalar(const Matrix& matrix, float value);

// --- elementwise math ---
Matrix exp(const Matrix& matrix);
Matrix log(const Matrix& matrix);
Matrix sqrt(const Matrix& matrix);
Matrix abs(const Matrix& matrix);
Matrix clamp(const Matrix& matrix, float low, float high);
Matrix sigmoid(const Matrix& matrix);
Matrix tanh(const Matrix& matrix);
Matrix leaky_relu(const Matrix& matrix, float negative_slope = 0.01f);
Matrix elu(const Matrix& matrix, float alpha = 1.0f);
Matrix gelu(const Matrix& matrix);
Matrix swish(const Matrix& matrix, float beta = 1.0f);
Matrix pow(const Matrix& matrix, float exponent);
Matrix negate(const Matrix& matrix);

// Training-oriented GPU kernels. Batch normalization normalizes each feature
// column; layer normalization normalizes each row.
Matrix batch_norm(const Matrix& matrix, float epsilon = 1e-5f);
Matrix layer_norm(const Matrix& matrix, float epsilon = 1e-5f);
Matrix dropout(const Matrix& matrix, float probability, unsigned long long seed = 0);

// NCHW convolution and pooling over rank-4 tensors.
Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias,
              std::size_t stride = 1, std::size_t padding = 0);
Tensor max_pool2d(const Tensor& input, std::size_t kernel_size,
                  std::size_t stride = 1, std::size_t padding = 0);
Tensor avg_pool2d(const Tensor& input, std::size_t kernel_size,
                  std::size_t stride = 1, std::size_t padding = 0);

// --- transforms ---
Matrix transpose(const Matrix& matrix);
Matrix relu(const Matrix& matrix);
Matrix softmax(const Matrix& matrix);
Matrix flatten(const Matrix& matrix);
Matrix slice(const Matrix& matrix, std::size_t row_start, std::size_t row_end,
	std::size_t col_start, std::size_t col_end);

// --- product variants ---
Matrix outer_product(const Matrix& left, const Matrix& right);
Matrix kron(const Matrix& left, const Matrix& right);

// Batched GEMM: a=[batch,m,k], b=[batch,k,n], result=[batch,m,n].
Tensor batch_matmul(const Tensor& left, const Tensor& right);

// --- broadcast ---
Matrix add_row_vector(const Matrix& matrix, const Matrix& vector);     // v length == cols
Matrix add_col_vector(const Matrix& matrix, const Matrix& vector);     // v length == rows
Matrix multiply_row_vector(const Matrix& matrix, const Matrix& vector);
Matrix multiply_col_vector(const Matrix& matrix, const Matrix& vector);

// --- axis reductions ---
Matrix row_sum(const Matrix& matrix);  // (rows x 1)
Matrix col_sum(const Matrix& matrix);  // (1 x cols)

// --- scalar reductions (GPU-accelerated) ---
float sum(const Matrix& matrix);
float mean(const Matrix& matrix);
float min(const Matrix& matrix);
float max(const Matrix& matrix);
std::size_t argmin(const Matrix& matrix);
std::size_t argmax(const Matrix& matrix);
float variance(const Matrix& matrix);
float stddev(const Matrix& matrix);
float l1_norm(const Matrix& matrix);
float l2_norm(const Matrix& matrix);
float frobenius_norm(const Matrix& matrix);
float abs_max(const Matrix& matrix);
float trace(const Matrix& matrix);
float determinant(const Matrix& matrix);
Matrix inverse(const Matrix& matrix);
float condition_number(const Matrix& matrix);
Matrix covariance(const Matrix& matrix);
Matrix correlation(const Matrix& matrix);

// --- comparison & logical ops ---
Matrix greater(const Matrix& matrix, float scalar);
Matrix greater(const Matrix& left, const Matrix& right);
Matrix less(const Matrix& matrix, float scalar);
Matrix less(const Matrix& left, const Matrix& right);
Matrix equal(const Matrix& matrix, float scalar);
Matrix equal(const Matrix& left, const Matrix& right);
Matrix not_equal(const Matrix& matrix, float scalar);
Matrix not_equal(const Matrix& left, const Matrix& right);
Matrix logical_and(const Matrix& left, const Matrix& right);
Matrix logical_or(const Matrix& left, const Matrix& right);
Matrix logical_not(const Matrix& matrix);
Matrix isnan(const Matrix& matrix);
Matrix isinf(const Matrix& matrix);
Matrix is_finite(const Matrix& matrix);
bool any(const Matrix& matrix);
bool all(const Matrix& matrix);
Matrix where(const Matrix& condition, const Matrix& true_value, const Matrix& false_value);
Matrix where(const Matrix& condition, float true_value, float false_value);
Matrix apply_mask(const Matrix& matrix, const Matrix& mask, float value);
Matrix filter_by_mask(const Matrix& matrix, const Matrix& mask);

// --- advanced linear algebra (cuSOLVER-accelerated) ---
// Solves A*x = b directly (getrf + getrs). A must be square; b may be a vector
// (n x 1) or a set of n right-hand sides (n x k). Returns a matrix shaped like b.
Matrix solve(const Matrix& left, const Matrix& rhs);
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
// Least-squares solution of A*x = b for possibly rank-deficient /
// rectangular systems, via the SVD pseudo-inverse.
Matrix solve_least_squares(const Matrix& left, const Matrix& rhs);

}


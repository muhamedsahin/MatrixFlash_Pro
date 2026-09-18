#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cusolverDn.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matrix_pro {
namespace {

inline void checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw SolverError(std::string(operation) + ": cusolver status " + std::to_string(static_cast<int>(status)));
    }
}

// Copy a row-major buffer into a column-major (Fortran) layout buffer of leading
// dimension `out_lda`. This is what cuSOLVER expects.
__global__ void row_major_to_column_major_kernel(const float* in, float* out,
                                                 std::size_t rows, std::size_t cols,
                                                 std::size_t out_lda) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const auto row = index / cols;
    const auto col = index % cols;
    out[col * out_lda + row] = in[row * cols + col];
}

// Copy a column-major buffer (leading dimension `in_lda`) into row-major output.
__global__ void column_major_to_row_major_kernel(const float* in, float* out,
                                                 std::size_t rows, std::size_t cols,
                                                 std::size_t in_lda) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const auto row = index / cols;
    const auto col = index % cols;
    out[row * cols + col] = in[col * in_lda + row];
}

// Extract the lower-triangular part of a column-major square matrix (lda=n) into a
// row-major square output, zero-filling the strict upper triangle.
__global__ void lower_triangular_cm_to_rm_kernel(const float* in, float* out, std::size_t n) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = n * n;
    if (index >= total) return;
    const auto row = index / n;
    const auto col = index % n;
    out[row * n + col] = (row >= col) ? in[col * n + row] : 0.0f;
}

// Extract the upper-triangular part (rows 0..k-1) of a column-major m x n buffer
// (lda=m) into a column-major k x n buffer (lda=k), zero-filling the lower part.
__global__ void upper_triangular_extract_kernel(const float* in, float* out,
                                                std::size_t m, std::size_t k, std::size_t n) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = k * n;
    if (index >= total) return;
    const auto row = index / n;
    const auto col = index % n;
    out[col * k + row] = (row <= col) ? in[col * m + row] : 0.0f;
}

} // namespace
Matrix solve(const Matrix& left, const Matrix& rhs) {
    if (left.rows() != left.cols()) {
        throw InvalidArgumentError("solve() requires a square coefficient matrix");
    }
    const std::size_t n = left.rows();
    const std::size_t k = rhs.cols();
    if (rhs.rows() != n) {
        throw ShapeMismatchError("solve() requires the RHS rows to match the coefficient matrix size");
    }
    if (n == 0) return Matrix(0, k);

    float* device_a = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    float* device_b = static_cast<float*>(allocate_device_memory(n * k * sizeof(float)));
    int* device_ipiv = static_cast<int*>(allocate_device_memory(n * sizeof(int)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            left.device_data(), device_a, n, n, n);
        checkCuda(cudaGetLastError(), "solve A conversion kernel");
        row_major_to_column_major_kernel<<<(n * k + 255) / 256, 256, 0, compute_stream()>>>(
            rhs.device_data(), device_b, n, k, n);
        checkCuda(cudaGetLastError(), "solve b conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSgetrf_bufferSize(cusolver_handle(), static_cast<int>(n),
                                                  static_cast<int>(n), device_a,
                                                  static_cast<int>(n), &lwork),
                      "solve workspace query");
        device_work = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSgetrf(cusolver_handle(), static_cast<int>(n), static_cast<int>(n),
                                       device_a, static_cast<int>(n), device_work, device_ipiv,
                                       device_info), "solve LU factorize");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "solve LU info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "solve LU info sync");
        if (info != 0) {
            throw MatrixProError("solve(): matrix is singular");
        }

        checkCusolver(cusolverDnSgetrs(cusolver_handle(), CUBLAS_OP_N, static_cast<int>(n),
                                       static_cast<int>(k), device_a, static_cast<int>(n),
                                       device_ipiv, device_b, static_cast<int>(n), device_info),
                      "solve back-substitution");
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "solve info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "solve info sync");
        if (info != 0) {
            throw MatrixProError("solve(): back-substitution failed");
        }

        Matrix output(n, k);
        column_major_to_row_major_kernel<<<(n * k + 255) / 256, 256, 0, compute_stream()>>>(
            device_b, output.device_data(), n, k, n);
        checkCuda(cudaGetLastError(), "solve output conversion kernel");

        free_device_memory(device_a);
        free_device_memory(device_b);
        free_device_memory(device_ipiv);
        free_device_memory(device_info);
        free_device_memory(device_work);
        output.mark_host_stale();
        return output;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_b);
        free_device_memory(device_ipiv);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}
QRResult qr(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0) {
        QRResult empty;
        empty.q = Matrix(m, 0);
        empty.r = Matrix(0, n);
        return empty;
    }
    const std::size_t k = std::min(m, n);

    float* device_a = static_cast<float*>(allocate_device_memory(m * n * sizeof(float)));
    float* device_r = static_cast<float*>(allocate_device_memory(k * n * sizeof(float)));
    float* device_tau = static_cast<float*>(allocate_device_memory(k * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_qr_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(m * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, m, n, m);
        checkCuda(cudaGetLastError(), "qr conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSgeqrf_bufferSize(cusolver_handle(), static_cast<int>(m),
                                                  static_cast<int>(n), device_a,
                                                  static_cast<int>(m), &lwork),
                      "qr geqrf workspace query");
        device_qr_work = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSgeqrf(cusolver_handle(), static_cast<int>(m), static_cast<int>(n),
                                       device_a, static_cast<int>(m), device_tau, device_qr_work,
                                       lwork, device_info), "qr geqrf");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "qr info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "qr info sync");
        if (info != 0) throw MatrixProError("qr(): factorization failed");

        // Preserve R before orgqr overwrites the leading columns with Q.
        upper_triangular_extract_kernel<<<(k * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, device_r, m, k, n);
        checkCuda(cudaGetLastError(), "qr R extraction kernel");

        checkCusolver(cusolverDnSorgqr_bufferSize(cusolver_handle(), static_cast<int>(m),
                                                  static_cast<int>(k), static_cast<int>(k),
                                                  device_a, static_cast<int>(m), device_tau,
                                                  &lwork), "qr orgqr workspace query");
        free_device_memory(device_qr_work);
        device_qr_work = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSorgqr(cusolver_handle(), static_cast<int>(m), static_cast<int>(k),
                                       static_cast<int>(k), device_a, static_cast<int>(m),
                                       device_tau, device_qr_work, lwork, device_info), "qr orgqr");
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "qr orgqr info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "qr orgqr info sync");
        if (info != 0) throw MatrixProError("qr(): Q generation failed");

        QRResult result;
        result.q = Matrix(m, k);
        result.r = Matrix(k, n);
        column_major_to_row_major_kernel<<<(m * k + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, result.q.device_data(), m, k, m);
        checkCuda(cudaGetLastError(), "qr Q conversion kernel");
        column_major_to_row_major_kernel<<<(k * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_r, result.r.device_data(), k, n, k);
        checkCuda(cudaGetLastError(), "qr R conversion kernel");

        free_device_memory(device_a);
        free_device_memory(device_r);
        free_device_memory(device_tau);
        free_device_memory(device_info);
        free_device_memory(device_qr_work);
        result.q.mark_host_stale();
        result.r.mark_host_stale();
        return result;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_r);
        free_device_memory(device_tau);
        free_device_memory(device_info);
        free_device_memory(device_qr_work);
        throw;
    }
}
SVDResult svd(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0) {
        SVDResult empty;
        empty.u = Matrix(m, m);
        empty.s = Matrix(0, 0);
        empty.v = Matrix(n, n);
        return empty;
    }
    const std::size_t minmn = std::min(m, n);

    float* device_a = static_cast<float*>(allocate_device_memory(m * n * sizeof(float)));
    float* device_s = static_cast<float*>(allocate_device_memory(minmn * sizeof(float)));
    float* device_u = static_cast<float*>(allocate_device_memory(m * m * sizeof(float)));
    float* device_vt = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    float* device_rwork = static_cast<float*>(allocate_device_memory(minmn * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(m * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, m, n, m);
        checkCuda(cudaGetLastError(), "svd conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSgesvd_bufferSize(cusolver_handle(), static_cast<int>(m),
                                                  static_cast<int>(n), &lwork),
                      "svd workspace query");
        device_work = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSgesvd(cusolver_handle(), 'A', 'A', static_cast<int>(m), static_cast<int>(n),
                                       device_a, static_cast<int>(m), device_s,
                                       device_u, static_cast<int>(m),
                                       device_vt, static_cast<int>(n),
                                       device_work, lwork, device_rwork, device_info), "svd gesvd");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "svd info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "svd info sync");
        if (info != 0) throw SolverError("svd(): singular value decomposition failed to converge");

        SVDResult result;
        result.u = Matrix(m, m);
        result.v = Matrix(n, n);
        column_major_to_row_major_kernel<<<(m * m + 255) / 256, 256, 0, compute_stream()>>>(
            device_u, result.u.device_data(), m, m, m);
        checkCuda(cudaGetLastError(), "svd U conversion kernel");
        column_major_to_row_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_vt, result.v.device_data(), n, n, n);
        checkCuda(cudaGetLastError(), "svd Vt conversion kernel");
        result.v = transpose(result.v);   // V = Vt^T

        std::vector<float> singular_values(minmn);
        checkCuda(cudaMemcpyAsync(singular_values.data(), device_s, minmn * sizeof(float),
                      cudaMemcpyDeviceToHost, compute_stream()), "svd singular values read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "svd singular values sync");
        std::vector<float> diag(minmn * minmn, 0.0f);
        for (std::size_t i = 0; i < minmn; ++i) diag[i * minmn + i] = singular_values[i];
        result.s = Matrix(minmn, minmn, diag);

        free_device_memory(device_a);
        free_device_memory(device_s);
        free_device_memory(device_u);
        free_device_memory(device_vt);
        free_device_memory(device_rwork);
        free_device_memory(device_info);
        free_device_memory(device_work);
        result.u.mark_host_stale();  // (result.v is marked by transpose above)
        return result;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_s);
        free_device_memory(device_u);
        free_device_memory(device_vt);
        free_device_memory(device_rwork);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}
Matrix cholesky(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw SolverError("cholesky() requires a square matrix");
    }
    const std::size_t n = matrix.rows();
    if (n == 0) return Matrix(0, 0);

    float* device_a = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, n, n, n);
        checkCuda(cudaGetLastError(), "cholesky conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSpotrf_bufferSize(cusolver_handle(), CUBLAS_FILL_MODE_LOWER,
                                                  static_cast<int>(n), device_a,
                                                  static_cast<int>(n), &lwork),
                      "cholesky workspace query");
        device_work = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSpotrf(cusolver_handle(), CUBLAS_FILL_MODE_LOWER, static_cast<int>(n),
                                       device_a, static_cast<int>(n), device_work, lwork,
                                       device_info), "cholesky potrf");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "cholesky info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "cholesky info sync");
        if (info != 0) {
            throw SolverError("cholesky(): matrix is not positive definite");
        }

        Matrix output(n, n);
        lower_triangular_cm_to_rm_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, output.device_data(), n);
        checkCuda(cudaGetLastError(), "cholesky output extraction kernel");

        free_device_memory(device_a);
        free_device_memory(device_info);
        free_device_memory(device_work);
        output.mark_host_stale();
        return output;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}

EigenResult eigen(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw SolverError("eigen() requires a square matrix");
    }
    const std::size_t n = matrix.rows();
    if (n == 0) {
        EigenResult empty;
        empty.eigenvalues = Matrix(0, 0);
        empty.eigenvectors = Matrix(0, 0);
        return empty;
    }

    float* device_a = static_cast<float*>(allocate_device_memory(n * n * sizeof(float)));
    float* device_w = static_cast<float*>(allocate_device_memory(n * sizeof(float)));
    int* device_info = static_cast<int*>(allocate_device_memory(sizeof(int)));
    float* device_work = nullptr;
    try {
        row_major_to_column_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            matrix.device_data(), device_a, n, n, n);
        checkCuda(cudaGetLastError(), "eigen conversion kernel");

        int lwork = 0;
        checkCusolver(cusolverDnSsyevd_bufferSize(cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_UPPER, static_cast<int>(n),
                                                  device_a, static_cast<int>(n), device_w, &lwork),
                      "eigen workspace query");
        device_work = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float)));

        checkCusolver(cusolverDnSsyevd(cusolver_handle(), CUSOLVER_EIG_MODE_VECTOR,
                                       CUBLAS_FILL_MODE_UPPER, static_cast<int>(n),
                                       device_a, static_cast<int>(n), device_w,
                                       device_work, lwork, device_info), "eigen syevd");
        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info, sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "eigen info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "eigen info sync");
        if (info != 0) {
            throw SolverError("eigen(): eigendecomposition failed (matrix may not be symmetric)");
        }

        EigenResult result;
        result.eigenvectors = Matrix(n, n);
        column_major_to_row_major_kernel<<<(n * n + 255) / 256, 256, 0, compute_stream()>>>(
            device_a, result.eigenvectors.device_data(), n, n, n);
        checkCuda(cudaGetLastError(), "eigen eigenvectors conversion kernel");

        std::vector<float> values(n);
        checkCuda(cudaMemcpyAsync(values.data(), device_w, n * sizeof(float),
                      cudaMemcpyDeviceToHost, compute_stream()), "eigen values read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "eigen values sync");
        std::vector<float> diag(n * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) diag[i * n + i] = values[i];
        result.eigenvalues = Matrix(n, n, diag);

        free_device_memory(device_a);
        free_device_memory(device_info);
        free_device_memory(device_work);
        result.eigenvectors.mark_host_stale();
        return result;
    } catch (...) {
        free_device_memory(device_a);
        free_device_memory(device_w);
        free_device_memory(device_info);
        free_device_memory(device_work);
        throw;
    }
}
Matrix pinv(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0) return Matrix(n, m);

    const SVDResult decomposition = svd(matrix);
    const std::size_t minmn = std::min(m, n);

    // Tolerance mirrors LAPACK's rcond-based truncation rule.
    float max_sigma = 0.0f;
    for (std::size_t i = 0; i < minmn; ++i) {
        max_sigma = std::max(max_sigma, decomposition.s.data()[i * minmn + i]);
    }
    const float tolerance = static_cast<float>(std::max(m, n)) *
                            std::numeric_limits<float>::epsilon() * max_sigma;

    std::vector<float> pseudo(n * m, 0.0f);   // n x m diagonal-ish inverse singular matrix
    for (std::size_t i = 0; i < minmn; ++i) {
        const float sigma = decomposition.s.data()[i * minmn + i];
        if (sigma > tolerance) pseudo[i * m + i] = 1.0f / sigma;
    }
    Matrix pseudo_matrix(n, m, pseudo);

    // A^+ = V * S^+ * U^T
    Matrix ut = transpose(decomposition.u);
    return multiply(multiply(decomposition.v, pseudo_matrix), ut);
}

std::size_t rank(const Matrix& matrix) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();
    if (m == 0 || n == 0) return 0;

    const SVDResult decomposition = svd(matrix);
    const std::size_t minmn = std::min(m, n);

    float max_sigma = 0.0f;
    for (std::size_t i = 0; i < minmn; ++i) {
        max_sigma = std::max(max_sigma, decomposition.s.data()[i * minmn + i]);
    }
    const float tolerance = static_cast<float>(std::max(m, n)) *
                            std::numeric_limits<float>::epsilon() * max_sigma;

    std::size_t count = 0;
    for (std::size_t i = 0; i < minmn; ++i) {
        if (decomposition.s.data()[i * minmn + i] > tolerance) ++count;
    }
    return count;
}

Matrix solve_least_squares(const Matrix& left, const Matrix& rhs) {
    if (rhs.rows() != left.rows()) {
        throw ShapeMismatchError("solve_least_squares() requires matching rows between A and b");
    }
    if (left.rows() == 0 || left.cols() == 0) return Matrix(left.cols(), rhs.cols());
    return multiply(pinv(left), rhs);
}

Matrix Matrix::solve(const Matrix& rhs) const { return matrix_pro::solve(*this, rhs); }
QRResult Matrix::qr() const { return matrix_pro::qr(*this); }
SVDResult Matrix::svd() const { return matrix_pro::svd(*this); }
Matrix Matrix::cholesky() const { return matrix_pro::cholesky(*this); }
EigenResult Matrix::eigen() const { return matrix_pro::eigen(*this); }
Matrix Matrix::pinv() const { return matrix_pro::pinv(*this); }
std::size_t Matrix::rank() const { return matrix_pro::rank(*this); }
Matrix Matrix::solve_least_squares(const Matrix& rhs) const { return matrix_pro::solve_least_squares(*this, rhs); }

} // namespace matrix_pro
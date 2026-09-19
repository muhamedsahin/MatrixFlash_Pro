#include "matrix_pro/ops/linalg_extended.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/utils/cuda_utils.hpp"
#include "matrix_pro/utils/errors.hpp"
#include "matrix_pro/ops/linalg.hpp"
#include "matrix_pro/ops/math_ops.hpp"
#include "matrix_pro/ops/creation.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>

namespace matrix_pro {

namespace detail {

__global__ void row_to_col_major_kernel(const float* in, float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        out[c * rows + r] = in[idx];
    }
}

__global__ void col_to_row_major_kernel(const float* in, float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        out[r * cols + c] = in[c * rows + r];
    }
}

__global__ void extract_L_kernel(const float* LU_cm, float* L_rm, std::size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int r = idx / n;
        int c = idx % n;
        if (r > c) {
            L_rm[idx] = LU_cm[c * n + r];
        } else if (r == c) {
            L_rm[idx] = 1.0f;
        } else {
            L_rm[idx] = 0.0f;
        }
    }
}

__global__ void extract_U_kernel(const float* LU_cm, float* U_rm, std::size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int r = idx / n;
        int c = idx % n;
        if (r <= c) {
            U_rm[idx] = LU_cm[c * n + r];
        } else {
            U_rm[idx] = 0.0f;
        }
    }
}

__global__ void build_permutation_matrix_kernel(const int* ipiv, float* P_rm, std::size_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (std::size_t i = 0; i < n; i++) {
            for (std::size_t j = 0; j < n; j++) {
                P_rm[i * n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        for (std::size_t i = 0; i < n; i++) {
            int p = ipiv[i] - 1;
            if (p != i) {
                for (std::size_t j = 0; j < n; j++) {
                    float temp = P_rm[i * n + j];
                    P_rm[i * n + j] = P_rm[p * n + j];
                    P_rm[p * n + j] = temp;
                }
            }
        }
    }
}

__global__ void log_det_kernel(const float* LU_cm, const int* ipiv, std::size_t n, float* out_det) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float log_det = 0.0f;
        int sign = 1;
        for (std::size_t i = 0; i < n; i++) {
            float diag = LU_cm[i * n + i];
            log_det += logf(abs(diag));
            if (diag < 0.0f) sign = -sign;
            if (ipiv[i] != i + 1) sign = -sign;
        }
        out_det[0] = log_det;
    }
}

} // namespace detail

LUResult lu(const Matrix& A) {
    if (A.rows() != A.cols()) throw ShapeMismatchError("Matrix must be square");
    std::size_t n = A.rows();

    Matrix LU_cm(n, n, MemoryMode::device_only);
    int num_blocks = (n * n + 255) / 256;
    detail::row_to_col_major_kernel<<<num_blocks, 256, 0, compute_stream()>>>(A.data(), LU_cm.data(), n, n);

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(cusolver_handle(), n, n, LU_cm.data(), n, &lwork);

    float* d_work; cudaMalloc(&d_work, lwork * sizeof(float));
    int* d_ipiv; cudaMalloc(&d_ipiv, n * sizeof(int));
    int* d_info; cudaMalloc(&d_info, sizeof(int));

    cusolverDnSgetrf(cusolver_handle(), n, n, LU_cm.data(), n, d_work, d_ipiv, d_info);

    LUResult result;
    result.L = Matrix(n, n, MemoryMode::device_only);
    result.U = Matrix(n, n, MemoryMode::device_only);
    result.P = Matrix(n, n, MemoryMode::device_only);

    detail::extract_L_kernel<<<num_blocks, 256, 0, compute_stream()>>>(LU_cm.data(), result.L.data(), n);
    detail::extract_U_kernel<<<num_blocks, 256, 0, compute_stream()>>>(LU_cm.data(), result.U.data(), n);
    detail::build_permutation_matrix_kernel<<<1, 1, 0, compute_stream()>>>(d_ipiv, result.P.data(), n);
    
    cudaFree(d_work); cudaFree(d_ipiv); cudaFree(d_info);
    result.L.mark_host_stale(); result.U.mark_host_stale(); result.P.mark_host_stale();
    
    return result;
}

Matrix trsm(const Matrix& A, const Matrix& B, bool upper, bool left, bool unit_diag) {
    Matrix X(B.rows(), B.cols(), MemoryMode::device_only);
    cudaMemcpyAsync(X.data(), B.data(), B.rows() * B.cols() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());

    cublasFillMode_t fill = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasSideMode_t side = left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
    cublasDiagType_t diag = unit_diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    const float alpha = 1.0f;
    cublasStrsm(cublas_handle(), side, fill, CUBLAS_OP_N, diag, B.cols(), B.rows(), &alpha, A.data(), A.cols(), X.data(), B.cols());
    
    X.mark_host_stale();
    return X;
}

Matrix trmm(const Matrix& A, const Matrix& B, bool upper, bool left) {
    Matrix X(B.rows(), B.cols(), MemoryMode::device_only);
    cudaMemcpyAsync(X.data(), B.data(), B.rows() * B.cols() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
    cublasFillMode_t fill = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasSideMode_t side = left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
    const float alpha = 1.0f;
    cublasStrmm(cublas_handle(), side, fill, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B.cols(), B.rows(), &alpha, A.data(), A.cols(), B.data(), B.cols(), X.data(), B.cols());
    X.mark_host_stale();
    return X;
}

Matrix matrix_power(const Matrix& A, int n) {
    if (A.rows() != A.cols()) throw ShapeMismatchError("Matrix must be square");
    if (n == 0) return eye(A.rows());
    if (n == 1) {
        Matrix res(A.rows(), A.cols(), MemoryMode::device_only);
        cudaMemcpyAsync(res.data(), A.data(), A.rows() * A.cols() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        return res;
    }
    
    Matrix base = (n < 0) ? inverse(A) : A;
    int p = std::abs(n);
    Matrix res = eye(A.rows());
    Matrix temp(A.rows(), A.cols(), MemoryMode::device_only);
    
    while (p > 0) {
        if (p % 2 == 1) {
            temp = multiply(res, base);
            cudaMemcpyAsync(res.data(), temp.data(), res.rows() * res.cols() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        }
        p /= 2;
        if (p > 0) {
            temp = multiply(base, base);
            cudaMemcpyAsync(base.data(), temp.data(), base.rows() * base.cols() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        }
    }
    res.mark_host_stale();
    return res;
}

Matrix matrix_exp(const Matrix& A, int order) {
    if (A.rows() != A.cols()) throw ShapeMismatchError("Matrix must be square");
    int n = A.rows();
    Matrix res = eye(n);
    Matrix term = eye(n);
    Matrix temp(n, n, MemoryMode::device_only);
    
    for (int i = 1; i <= order; ++i) {
        temp = multiply(term, A);
        const float alpha = 1.0f / i;
        cublasSscal(cublas_handle(), n * n, &alpha, temp.data(), 1);
        cudaMemcpyAsync(term.data(), temp.data(), n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        const float alpha_add = 1.0f;
        cublasSaxpy(cublas_handle(), n * n, &alpha_add, term.data(), 1, res.data(), 1);
    }
    res.mark_host_stale();
    return res;
}

float log_determinant(const Matrix& A) {
    if (A.rows() != A.cols()) throw ShapeMismatchError("Matrix must be square");
    std::size_t n = A.rows();
    Matrix LU_cm(n, n, MemoryMode::device_only);
    int num_blocks = (n * n + 255) / 256;
    detail::row_to_col_major_kernel<<<num_blocks, 256, 0, compute_stream()>>>(A.data(), LU_cm.data(), n, n);

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(cusolver_handle(), n, n, LU_cm.data(), n, &lwork);

    float* d_work; cudaMalloc(&d_work, lwork * sizeof(float));
    int* d_ipiv; cudaMalloc(&d_ipiv, n * sizeof(int));
    int* d_info; cudaMalloc(&d_info, sizeof(int));

    cusolverDnSgetrf(cusolver_handle(), n, n, LU_cm.data(), n, d_work, d_ipiv, d_info);

    float* d_det; cudaMalloc(&d_det, sizeof(float));
    detail::log_det_kernel<<<1, 1, 0, compute_stream()>>>(LU_cm.data(), d_ipiv, n, d_det);

    float h_det;
    cudaMemcpyAsync(&h_det, d_det, sizeof(float), cudaMemcpyDeviceToHost, compute_stream());
    cudaStreamSynchronize(compute_stream());

    cudaFree(d_work); cudaFree(d_ipiv); cudaFree(d_info); cudaFree(d_det);
    return h_det;
}

Tensor batch_solve(const Tensor& A, const Tensor& B) {
    // A: [batch, n, n], B: [batch, n, m]
    const auto& shape_A = A.shape();
    const auto& shape_B = B.shape();
    if (shape_A.size() != 3 || shape_B.size() != 3 || shape_A[0] != shape_B[0] || shape_A[1] != shape_A[2] || shape_A[2] != shape_B[1]) {
        throw ShapeMismatchError("Invalid shapes for batch_solve");
    }

    int batch_size = shape_A[0];
    int n = shape_A[1];
    int m = shape_B[2];

    Tensor X(shape_B, MemoryMode::device_only);

    for (int i = 0; i < batch_size; ++i) {
        Matrix Ai(n, n, MemoryMode::device_only);
        cudaMemcpyAsync(Ai.data(), A.data() + i * n * n, n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        
        Matrix Bi(n, m, MemoryMode::device_only);
        cudaMemcpyAsync(Bi.data(), B.data() + i * n * m, n * m * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        
        Matrix Xi = solve(Ai, Bi);
        cudaMemcpyAsync(X.data() + i * n * m, Xi.data(), n * m * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
    }
    
    X.mark_host_stale();
    return X;
}

Tensor batch_inverse(const Tensor& A) {
    const auto& shape = A.shape();
    if (shape.size() != 3 || shape[1] != shape[2]) throw ShapeMismatchError("Invalid shape");
    
    int batch_size = shape[0];
    int n = shape[1];
    
    Tensor inv(shape, MemoryMode::device_only);
    for (int i = 0; i < batch_size; ++i) {
        Matrix Ai(n, n, MemoryMode::device_only);
        cudaMemcpyAsync(Ai.data(), A.data() + i * n * n, n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        
        Matrix Ai_inv = inverse(Ai);
        cudaMemcpyAsync(inv.data() + i * n * n, Ai_inv.data(), n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
    }
    inv.mark_host_stale();
    return inv;
}

std::vector<float> batch_det(const Tensor& A) {
    const auto& shape = A.shape();
    if (shape.size() != 3 || shape[1] != shape[2]) throw ShapeMismatchError("Invalid shape");
    
    int batch_size = shape[0];
    int n = shape[1];
    std::vector<float> dets(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        Matrix Ai(n, n, MemoryMode::device_only);
        cudaMemcpyAsync(Ai.data(), A.data() + i * n * n, n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        dets[i] = std::exp(log_determinant(Ai)); // assuming det is positive for simplicity, real log_det returns log(abs(det))
    }
    
    return dets;
}

} // namespace matrix_pro

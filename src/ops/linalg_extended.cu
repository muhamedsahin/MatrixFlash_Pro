#include "matrix_pro/ops/linalg_extended.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/ops/linalg.hpp"
#include "matrix_pro/ops/operations.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>
#include <limits>
#include "matrix_pro/ops/gemm.hpp"
#include "matrix_pro/ops/factorization.hpp"

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
            log_det += logf(fabsf(diag));
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
    detail::row_to_col_major_kernel<<<num_blocks, 256, 0, compute_stream()>>>(A.device_data(), LU_cm.device_data(), n, n);

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(cusolver_handle(), n, n, LU_cm.device_data(), n, &lwork);

    float* d_work; cudaMalloc(&d_work, lwork * sizeof(float));
    int* d_ipiv; cudaMalloc(&d_ipiv, n * sizeof(int));
    int* d_info; cudaMalloc(&d_info, sizeof(int));

    cusolverDnSgetrf(cusolver_handle(), n, n, LU_cm.device_data(), n, d_work, d_ipiv, d_info);

    LUResult result;
    result.L = Matrix(n, n, MemoryMode::device_only);
    result.U = Matrix(n, n, MemoryMode::device_only);
    result.P = Matrix(n, n, MemoryMode::device_only);

    detail::extract_L_kernel<<<num_blocks, 256, 0, compute_stream()>>>(LU_cm.device_data(), result.L.device_data(), n);
    detail::extract_U_kernel<<<num_blocks, 256, 0, compute_stream()>>>(LU_cm.device_data(), result.U.device_data(), n);
    detail::build_permutation_matrix_kernel<<<1, 1, 0, compute_stream()>>>(d_ipiv, result.P.device_data(), n);
    
    cudaFree(d_work); cudaFree(d_ipiv); cudaFree(d_info);
    result.L.mark_host_stale(); result.U.mark_host_stale(); result.P.mark_host_stale();
    
    return result;
}

Matrix trsm(const Matrix& A, const Matrix& B, bool upper, bool left, bool unit_diag) {
    Matrix X(B.rows(), B.cols(), MemoryMode::device_only);
    cudaMemcpyAsync(X.device_data(), B.device_data(), B.rows() * B.cols() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());

    cublasFillMode_t fill = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasSideMode_t side = left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
    cublasDiagType_t diag = unit_diag ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    const float alpha = 1.0f;
    cublasStrsm(cublas_handle(), side, fill, CUBLAS_OP_N, diag, B.cols(), B.rows(), &alpha, A.device_data(), A.cols(), X.device_data(), B.cols());
    
    X.mark_host_stale();
    return X;
}

Matrix trmm(const Matrix& A, const Matrix& B, bool upper, bool left) {
    Matrix X(B.rows(), B.cols(), MemoryMode::device_only);
    cudaMemcpyAsync(X.device_data(), B.device_data(), B.rows() * B.cols() * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
    cublasFillMode_t fill = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cublasSideMode_t side = left ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
    const float alpha = 1.0f;
    cublasStrmm(cublas_handle(), side, fill, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B.cols(), B.rows(), &alpha, A.device_data(), A.cols(), B.device_data(), B.cols(), X.device_data(), B.cols());
    X.mark_host_stale();
    return X;
}

Matrix matrix_power(const Matrix& A, int exponent) {
    if(A.rows()!=A.cols())throw ShapeMismatchError("matrix_power requires square input");
    std::uint64_t power=exponent<0?static_cast<std::uint64_t>(-static_cast<std::int64_t>(exponent)):static_cast<std::uint64_t>(exponent);
    Matrix result=Matrix::identity(A.rows());
    if(!power)return result;
    Matrix base=exponent<0?inverse(A):Matrix(A);
    while(power) {
        if(power&1)result=gemm(result,base);
        power>>=1;
        if(power)base=gemm(base,base);
    }
    return result;
}

Matrix matrix_exp(const Matrix& A, int order) {
    if(A.rows()!=A.cols())throw ShapeMismatchError("matrix_exp requires square input");
    if(order<1||order>64)throw InvalidArgumentError("matrix_exp order must be in [1,64]");
    if(A.empty())return Matrix(0,0,MemoryMode::device_only);
    // Infinity norm bounds the spectral radius and governs scaling. FP32 GEMM
    // forbids implicit TF32 input conversion throughout this scientific path.
    const float norm=matrix_pro::abs(A).row_sum().max();
    if(!std::isfinite(norm))throw InvalidArgumentError("matrix_exp requires finite input and norm");
    const double threshold=order==13?4.25:0.5;
    const int squarings=norm>threshold?static_cast<int>(std::ceil(std::log2(norm/threshold))):0;
    Matrix a=A*std::ldexp(1.0f,-squarings),identity=Matrix::identity(A.rows()),result;
    if(order==13) {
        // [13/13] Pade; coefficients normalized by b0 keep FP32 intermediates
        // away from the huge unnormalized coefficients (~6.5e16).
        const double coefficients[]={64764752532480000.,32382376266240000.,7771770303897600.,
            1187353796428800.,129060195264000.,10559470521600.,670442572800.,33522128640.,
            1323241920.,40840800.,960960.,16380.,182.,1.};
        float b[14];for(int i=0;i<14;++i)b[i]=static_cast<float>(coefficients[i]/coefficients[0]);
        Matrix a2=gemm(a,a),a4=gemm(a2,a2),a6=gemm(a4,a2);
        Matrix u=gemm(a,gemm(a6,a6*b[13]+a4*b[11]+a2*b[9])+a6*b[7]+a4*b[5]+a2*b[3]+identity*b[1]);
        Matrix v=gemm(a6,a6*b[12]+a4*b[10]+a2*b[8])+a6*b[6]+a4*b[4]+a2*b[2]+identity;
        LUFactorization denominator(v-u,A.cols());
        result=denominator.solve(v+u);
    } else {
        result=identity;Matrix term=identity;
        for(int i=1;i<=order;++i) {term=gemm(term,a)*(1.0f/i);result=result+term;}
    }
    for(int i=0;i<squarings;++i)result=gemm(result,result);
    result.mark_host_stale();return result;
}

float log_determinant(const Matrix& A) {
    return static_cast<float>(slogdet(A).log_abs_det);
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
        cudaMemcpyAsync(Ai.device_data(), A.device_data() + i * n * n, n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        
        Matrix Bi(n, m, MemoryMode::device_only);
        cudaMemcpyAsync(Bi.device_data(), B.device_data() + i * n * m, n * m * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        
        Matrix Xi = solve(Ai, Bi);
        cudaMemcpyAsync(X.device_data() + i * n * m, Xi.device_data(), n * m * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
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
        cudaMemcpyAsync(Ai.device_data(), A.device_data() + i * n * n, n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        
        Matrix Ai_inv = inverse(Ai);
        cudaMemcpyAsync(inv.device_data() + i * n * n, Ai_inv.device_data(), n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
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
        cudaMemcpyAsync(Ai.device_data(), A.device_data() + i * n * n, n * n * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream());
        dets[i] = determinant(Ai);
    }
    
    return dets;
}

} // namespace matrix_pro

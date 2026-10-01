// Dahili uygulama parçası: src/ops/linalg_extended.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: LU, permutation ve log-determinant yardımcı çekirdekleri.

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
            if (diag < 0.0f)
                sign = -sign;
            if (ipiv[i] != i + 1)
                sign = -sign;
        }
        out_det[0] = log_det;
    }
}

} // namespace detail

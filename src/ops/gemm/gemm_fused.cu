#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>

#include <mutex>

namespace matrix_pro {
namespace detail {
namespace gemm {
namespace {

__global__ void epilogue_col_bias_kernel(float* __restrict__ c,
                                         const float* __restrict__ bias,
                                         int m, int n, int mode) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(m) * n;
    if (index >= total) return;
    const int col = static_cast<int>(index % static_cast<std::size_t>(n));
    float v = c[index] + bias[col];
    if (mode == 1) {
        v = v > 0.0f ? v : 0.0f;
    } else if (mode == 2) {
        const float x3 = v * v * v;
        v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * x3)));
    }
    c[index] = v;
}

struct FusedLt {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 64u * 1024u * 1024u;
    std::mutex mutex;

    FusedLt() {
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            handle = nullptr;
            return;
        }
        if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
            cudaGetLastError();
            workspace = nullptr;
            workspace_bytes = 0;
        }
    }
    ~FusedLt() {
        if (workspace != nullptr) cudaFree(workspace);
        if (handle != nullptr) cublasLtDestroy(handle);
    }
    static FusedLt& instance() {
        static FusedLt s;
        return s;
    }
};

cublasLtEpilogue_t to_lt_epilogue(Epilogue epi) {
    switch (epi) {
        case Epilogue::bias: return CUBLASLT_EPILOGUE_BIAS;
        case Epilogue::bias_relu: return CUBLASLT_EPILOGUE_RELU_BIAS;
        case Epilogue::bias_gelu: return CUBLASLT_EPILOGUE_GELU_BIAS;
        default: return CUBLASLT_EPILOGUE_DEFAULT;
    }
}

bool try_lt_fused(const float* a, const float* b, float* c, const float* bias,
                  int m, int n, int k, Epilogue epi, cudaStream_t stream) {
    if (epi == Epilogue::none || bias == nullptr) return false;
    FusedLt& lt = FusedLt::instance();
    if (lt.handle == nullptr) return false;

    std::lock_guard<std::mutex> lock(lt.mutex);
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t adesc = nullptr;
    cublasLtMatrixLayout_t bdesc = nullptr;
    cublasLtMatrixLayout_t cdesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    const cublasLtEpilogue_t lt_epi = to_lt_epilogue(epi);
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &lt_epi, sizeof(lt_epi)) !=
        CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(op_desc);
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias, sizeof(bias)) !=
        CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(op_desc);
        return false;
    }

    cublasLtMatrixLayoutCreate(&adesc, CUDA_R_32F, n, k, n);
    cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_32F, k, m, k);
    cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32F, n, m, n);
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt.workspace_bytes,
        sizeof(lt.workspace_bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
        lt.handle, op_desc, &alpha, b, adesc, a, bdesc, &beta, c, cdesc, c,
        cdesc, nullptr, lt.workspace, lt.workspace_bytes, stream);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(cdesc);
    cublasLtMatrixLayoutDestroy(bdesc);
    cublasLtMatrixLayoutDestroy(adesc);
    cublasLtMatmulDescDestroy(op_desc);
    return status == CUBLAS_STATUS_SUCCESS;
}

} // namespace

void apply_epilogue(float* c, const float* bias, int m, int n, Epilogue epi,
                    cudaStream_t stream) {
    if (epi == Epilogue::none || bias == nullptr || m <= 0 || n <= 0) return;
    int mode = 0;
    if (epi == Epilogue::bias_relu) mode = 1;
    else if (epi == Epilogue::bias_gelu) mode = 2;
    const std::size_t total = static_cast<std::size_t>(m) * n;
    const unsigned block = 256;
    const unsigned grid =
        static_cast<unsigned>((total + block - 1) / block);
    epilogue_col_bias_kernel<<<grid, block, 0, stream>>>(c, bias, m, n, mode);
    checkCuda(cudaGetLastError(), "epilogue kernel launch");
}

void gemm_bias_epilogue(const float* a, const float* b, float* c,
                        const float* bias, int m, int n, int k, Epilogue epi,
                        cudaStream_t stream) {
    if (try_lt_fused(a, b, c, bias, m, n, k, epi, stream)) return;
    gemm_rowmajor(a, b, c, m, n, k, stream);
    apply_epilogue(c, bias, m, n, epi, stream);
}

} // namespace gemm
} // namespace detail
} // namespace matrix_pro

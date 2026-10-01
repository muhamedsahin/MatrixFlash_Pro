// Dahili uygulama parçası: src/ops/gemm/gemm_plan.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Vendor/fallback epilogue seçimi ve sıfır-K output ölçekleme.

cublasLtEpilogue_t lt_epilogue(GemmEpilogue e) {
    switch (e) {
    case GemmEpilogue::bias:
        return CUBLASLT_EPILOGUE_BIAS;
    case GemmEpilogue::bias_relu:
        return CUBLASLT_EPILOGUE_RELU_BIAS;
    case GemmEpilogue::bias_gelu:
        return CUBLASLT_EPILOGUE_GELU_BIAS;
    default:
        return CUBLASLT_EPILOGUE_DEFAULT;
    }
}
detail::gemm::Epilogue fallback_epilogue(GemmEpilogue e) {
    return static_cast<detail::gemm::Epilogue>(static_cast<int>(e));
}
__global__ void scale_output(float* output, std::size_t count, float beta) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        output[i] = beta == 0 ? 0 : beta * output[i];
}

// Dahili uygulama parçası: src/core/cuda_utils.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Hata çevirme, compute stream, SM sayısı ve cuBLAS erişimi.

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw CudaError(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void synchronize() {
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");
}

int sm_count() {
    int device = 0;
    cudaGetDevice(&device);
    static thread_local std::unordered_map<int, int> cached;
    const auto it = cached.find(device);
    if (it != cached.end())
        return it->second;
    int sms = 1;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        cudaGetLastError();
        sms = 1;
    }
    return cached.emplace(device, sms).first->second;
}

cudaStream_t compute_stream() {
    auto& context = execution_context();
    return context.override_stream ? context.override_stream : context.stream;
}
namespace detail {
void set_compute_stream(cudaStream_t stream) {
    auto& context = execution_context();
    context.override_stream = stream == context.stream ? nullptr : stream;
}
}
cublasHandle_t& cublas_handle() {
    auto& context = execution_context();
    const auto stream = compute_stream();
    if (context.blas_stream != stream) {
        if (cublasSetStream(context.blas, stream) != CUBLAS_STATUS_SUCCESS)
            throw CudaError("cublasSetStream failed");
        // cublasSetStream resets the workspace. On alternate streams let cuBLAS
        // own its workspace so pending default-stream work cannot race with it.
        if (stream == context.stream && context.blas_workspace)
            cublasSetWorkspace(context.blas, context.blas_workspace, context.workspace_bytes);
        context.blas_stream = stream;
    }
    return context.blas;
}

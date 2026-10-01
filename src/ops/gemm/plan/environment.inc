// Dahili uygulama parçası: src/ops/gemm/gemm_plan.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Thread/cihaz/stream başına vendor handle ve lazy workspace.

void check_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw CudaError(std::string(operation) + ": cuBLAS status " + std::to_string(status));
}
void require_not_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    checkCuda(cudaStreamIsCapturing(stream, &status), "GEMM capture query");
    if (status != cudaStreamCaptureStatusNone)
        throw InvalidArgumentError("Construct/tune GEMM plans before CUDA Graph capture");
}
int checked_dim(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw InvalidArgumentError("GEMM dimension exceeds the 32-bit backend limit");
    return static_cast<int>(value);
}
// Shared only within one thread/device/stream. Scratch memory is never used by
// independent streams, even when their submissions come from the same thread.
struct Environment {
    int device;
    cudaStream_t stream;
    cublasLtHandle_t lt = nullptr;
    cublasHandle_t blas = nullptr;
    void* workspace = nullptr;
    std::size_t bytes = 64u << 20;
    Environment(int d, cudaStream_t s) : device(d), stream(s) {
        try {
            check_blas(cublasCreate(&blas), "plan cuBLAS create");
            check_blas(cublasSetStream(blas, stream), "plan cuBLAS stream");
            check_blas(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "plan scalar mode");
            if (cudaMalloc(&workspace, bytes) != cudaSuccess) {
                cudaGetLastError();
                bytes = 0;
                workspace = nullptr;
            }
            if (workspace)
                check_blas(cublasSetWorkspace(blas, workspace, bytes), "plan workspace");
            if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS)
                lt = nullptr;
        } catch (...) {
            release();
            throw;
        }
    }
    void release() noexcept {
        int old = 0;
        cudaGetDevice(&old);
        if (old != device)
            cudaSetDevice(device);
        // cudaFree waits for users of scratch; ensure handles outlive submissions.
        if (workspace)
            cudaFree(workspace);
        if (lt)
            cublasLtDestroy(lt);
        if (blas)
            cublasDestroy(blas);
        workspace = nullptr;
        lt = nullptr;
        blas = nullptr;
        if (old != device)
            cudaSetDevice(old);
    }
    ~Environment() { release(); }
};
std::shared_ptr<Environment> environment(int device, cudaStream_t stream) {
    using Key = std::pair<int, cudaStream_t>;
    static thread_local std::map<Key, std::weak_ptr<Environment>> states;
    const Key key{device, stream};
    auto it = states.find(key);
    if (it != states.end())
        if (auto existing = it->second.lock())
            return existing;
    for (auto entry = states.begin(); entry != states.end();) {
        if (entry->second.expired())
            entry = states.erase(entry);
        else
            ++entry;
    }
    auto created = std::make_shared<Environment>(device, stream);
    states[key] = created;
    return created;
}

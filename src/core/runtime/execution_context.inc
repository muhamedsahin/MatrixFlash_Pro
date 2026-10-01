// Dahili uygulama parçası: src/core/cuda_utils.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Thread ve cihaz başına stream/cuBLAS; kalıcı 64 MiB workspace korunur.

struct CudaExecutionContext {
    int device = 0;
    cudaStream_t stream = nullptr;
    cudaStream_t override_stream = nullptr;
    cudaStream_t blas_stream = nullptr;
    cublasHandle_t blas = nullptr;
    // Persistent cuBLAS workspace: without it cuBLAS allocates/frees an
    // internal scratch buffer on the first GEMM of every new shape, which
    // serializes the stream and fights our own memory pool. One 64 MiB
    // workspace per context removes that cost entirely.
    static constexpr std::size_t workspace_bytes = 64u * 1024u * 1024u;
    void* blas_workspace = nullptr;

    explicit CudaExecutionContext(int device_index) : device(device_index) {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            throw CudaError("cudaStreamCreateWithFlags failed");
        }
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            cudaStreamDestroy(stream);
            throw CudaError("cublasCreate failed");
        }
        if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cudaStreamDestroy(stream);
            throw CudaError("cublasSetStream failed");
        }
        blas_stream = stream;
        if (cudaMalloc(&blas_workspace, workspace_bytes) == cudaSuccess) {
            if (cublasSetWorkspace(blas, blas_workspace, workspace_bytes) !=
                CUBLAS_STATUS_SUCCESS) {
                cublasSetWorkspace(blas, nullptr, 0);
                cudaFree(blas_workspace);
                blas_workspace = nullptr;
            }
        } else {
            cudaGetLastError(); // clear the sticky error, run without a workspace
        }
        // Allow cuBLAS to use atomic-based split-k reductions; on tall/skinny
        // GEMMs this is measurably faster and only changes accumulation order.
        cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_ALLOWED);
        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, device);
        if (properties.major >= 8)
            cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH);
    }

    ~CudaExecutionContext() {
        int old = 0;
        cudaGetDevice(&old);
        if (old != device)
            cudaSetDevice(device);
        if (blas != nullptr) {
            cublasSetWorkspace(blas, nullptr, 0);
            cublasDestroy(blas);
        }
        if (blas_workspace != nullptr)
            cudaFree(blas_workspace);
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        if (old != device)
            cudaSetDevice(old);
    }
};

// One execution context per (host thread, device). A thread that never calls
// into the library allocates nothing, and two threads never share a stream, so
// concurrent use from independent host threads is safe by construction.
struct PerThreadContexts {
    std::unordered_map<int, std::unique_ptr<CudaExecutionContext>> by_device;

    CudaExecutionContext& for_device(int device) {
        auto it = by_device.find(device);
        if (it != by_device.end())
            return *it->second;
        auto context = std::make_unique<CudaExecutionContext>(device);
        CudaExecutionContext& reference = *context;
        by_device.emplace(device, std::move(context));
        return reference;
    }

    ~PerThreadContexts() { by_device.clear(); }
};

PerThreadContexts& thread_contexts() {
    static thread_local PerThreadContexts contexts;
    return contexts;
}

CudaExecutionContext& execution_context() {
    int device = 0;
    cudaGetDevice(&device);
    return thread_contexts().for_device(device);
}

// Dahili uygulama parçası: src/core/cuda_utils.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: cuSOLVER ve isteğe bağlı cuDNN handle önbelleği.

// cuSOLVER/cuDNN handles per (thread, device). Lazily created and reused so
// repeated decompositions/convolutions do not pay handle-construction cost,
// while different threads and devices stay isolated from one another.
struct PerThreadSolvers {
    std::unordered_map<int, cusolverDnHandle_t> solver_by_device;
#ifdef MATRIX_PRO_HAS_CUDNN
    std::unordered_map<int, cudnnHandle_t> cudnn_by_device;
#endif

    cusolverDnHandle_t& solver_for(int device) {
        auto it = solver_by_device.find(device);
        if (it != solver_by_device.end())
            return it->second;
        cusolverDnHandle_t handle = nullptr;
        if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
            throw SolverError("cuSOLVER handle creation failed");
        }
        if (cusolverDnSetStream(handle, execution_context().stream) != CUSOLVER_STATUS_SUCCESS) {
            cusolverDnDestroy(handle);
            throw SolverError("cusolverDnSetStream failed");
        }
        return solver_by_device.emplace(device, handle).first->second;
    }

#ifdef MATRIX_PRO_HAS_CUDNN
    cudnnHandle_t& cudnn_for(int device) {
        auto it = cudnn_by_device.find(device);
        if (it != cudnn_by_device.end())
            return it->second;
        cudnnHandle_t handle = nullptr;
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
            throw CudaError("cudnnCreate failed");
        }
        if (cudnnSetStream(handle, execution_context().stream) != CUDNN_STATUS_SUCCESS) {
            cudnnDestroy(handle);
            throw CudaError("cudnnSetStream failed");
        }
        return cudnn_by_device.emplace(device, handle).first->second;
    }
#endif

    ~PerThreadSolvers() {
        for (auto& entry : solver_by_device) {
            if (entry.second != nullptr)
                cusolverDnDestroy(entry.second);
        }
#ifdef MATRIX_PRO_HAS_CUDNN
        for (auto& entry : cudnn_by_device) {
            if (entry.second != nullptr)
                cudnnDestroy(entry.second);
        }
#endif
    }
};

PerThreadSolvers& thread_solvers() {
    static thread_local PerThreadSolvers solvers;
    return solvers;
}

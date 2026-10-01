// Dahili uygulama parçası: src/core/cuda_utils.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Havuzdan tahsis/iade; farklı stream yeniden kullanımında event beklenir.

void* allocate_device_memory(std::size_t bytes) {
    if (bytes == 0)
        return nullptr;
    int device = 0;
    cudaGetDevice(&device);
    auto& pool = DeviceMemoryPool::instance();
    std::lock_guard<std::mutex> lock(pool.mutex);

    const std::size_t cl = size_class(bytes);
    auto& bucket = pool.free_blocks[device][cl];
    if (!bucket.empty()) {
        void* pointer = bucket.back();
        bucket.pop_back();
        const cudaEvent_t ready = pool.block_info[pointer].ready_event;
        pool.block_info[pointer] = DeviceMemoryPool::BlockInfo{device, cl, nullptr};
        if (ready != nullptr) {
            // Cross-stream reuse: make this stream wait until the stream that
            // freed the block (and therefore all GPU work touching it) has
            // reached the recycle point. Same-stream reuse is already ordered
            // by the stream itself, so this only fires across streams.
            const cudaError_t wait_status = cudaStreamWaitEvent(compute_stream(), ready, 0);
            cudaEventDestroy(ready);
            if (wait_status != cudaSuccess) {
                // Could not establish ordering: releasing the block is the
                // only safe fallback (cudaFree implicitly synchronizes).
                pool.block_info.erase(pointer);
                cudaFree(pointer);
                throw CudaError("cudaStreamWaitEvent failed for a recycled memory-pool block");
            }
        }
        return pointer;
    }

    void* pointer = nullptr;
    if (cudaMalloc(&pointer, cl) != cudaSuccess) {
        throw CudaError("cudaMalloc failed in device memory pool");
    }
    pool.block_info[pointer] = DeviceMemoryPool::BlockInfo{device, cl};
    return pointer;
}

void free_device_memory(void* pointer) {
    if (pointer == nullptr)
        return;
    auto& pool = DeviceMemoryPool::instance();
    std::lock_guard<std::mutex> lock(pool.mutex);

    auto info_it = pool.block_info.find(pointer);
    if (info_it == pool.block_info.end()) {
        cudaFree(pointer);
        return;
    }

    // Recycle into the bucket of the device the block was allocated on, never
    // into another device's bucket.
    const int device = info_it->second.device;
    const std::size_t cl = info_it->second.size_class;
    pool.block_info.erase(info_it);
    auto& bucket = pool.free_blocks[device][cl];
    if (bucket.size() < DeviceMemoryPool::max_reuse_per_size) {
        // Record an event on the current thread's compute stream so a later
        // allocation on a DIFFERENT stream waits for pending GPU work that
        // touched this block before reusing it (see allocate_device_memory).
        // If recording fails for any reason, a plain cudaFree is always safe.
        cudaEvent_t ready = nullptr;
        bool cached = false;
        try {
            if (cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) == cudaSuccess) {
                if (cudaEventRecord(ready, compute_stream()) == cudaSuccess) {
                    pool.block_info[pointer] = DeviceMemoryPool::BlockInfo{device, cl, ready};
                    bucket.push_back(pointer);
                    cached = true;
                }
            }
        } catch (...) {
            // free_device_memory runs from destructors (RAII guards, Matrix) --
            // it must never throw. Fall through to the unconditional free.
        }
        if (cached)
            return;
        if (ready != nullptr)
            cudaEventDestroy(ready);
        cudaFree(pointer);
        return;
    }
    cudaFree(pointer);
}

void zero_device_memory(void* pointer, std::size_t bytes) {
    if (pointer == nullptr || bytes == 0)
        return;
    checkCuda(cudaMemsetAsync(pointer, 0, bytes, compute_stream()),
              "cudaMemsetAsync in zero_device_memory");
}

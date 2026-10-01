// Dahili uygulama parçası: src/core/cuda_utils.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Tek sahiplik haritası, cihaz etiketleri ve power-of-two sınıfları.

struct DeviceMemoryPool {
    static constexpr std::size_t max_reuse_per_size = 32;

    struct BlockInfo {
        int device = 0;
        std::size_t size_class = 0;
        // Recorded when the block was recycled (see free_device_memory): a
        // future allocation on a different stream waits on it before reuse.
        cudaEvent_t ready_event = nullptr;
    };

    std::mutex mutex;
    // device -> size class -> reusable blocks
    std::unordered_map<int, std::unordered_map<std::size_t, std::vector<void*>>> free_blocks;
    std::unordered_map<void*, BlockInfo> block_info;

    ~DeviceMemoryPool() {
        // free_blocks is only an index into block_info: freeing both collections
        // double-frees every cached allocation at shutdown. The ownership map
        // contains cached and outstanding blocks, so release each exactly once.
        int old = 0;
        cudaGetDevice(&old);
        for (auto& entry : block_info) {
            cudaSetDevice(entry.second.device);
            if (entry.first != nullptr)
                cudaFree(entry.first);
            if (entry.second.ready_event != nullptr)
                cudaEventDestroy(entry.second.ready_event);
        }
        cudaSetDevice(old);
    }

    static DeviceMemoryPool& instance() {
        static DeviceMemoryPool pool;
        return pool;
    }
};

// Power-of-two size classes: requests are rounded up so that freed blocks are
// reusable for any nearby request size, avoiding per-exact-size fragmentation.
std::size_t size_class(std::size_t bytes) {
    constexpr std::size_t min_class = 256;
    if (bytes > std::numeric_limits<std::size_t>::max() / 2 + 1)
        throw InvalidArgumentError("Device allocation size exceeds the pool size-class limit");
    std::size_t cl = min_class;
    while (cl < bytes)
        cl *= 2;
    return cl;
}

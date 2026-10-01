// Dahili uygulama parçası: src/core/cuda_utils.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Cihaz seçimi ve DeviceScope kapsam sonunda eski cihazı geri yükler.

int device_count() {
    static const int count = [] {
        int detected = 0;
        if (cudaGetDeviceCount(&detected) != cudaSuccess)
            return 0;
        return detected;
    }();
    return count;
}

int current_device() {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

void set_device(int device) {
    if (device < 0 || device >= device_count()) {
        throw InvalidArgumentError("Invalid CUDA device index");
    }
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
}

void synchronize_all_devices() {
    for (auto& entry : thread_contexts().by_device) {
        if (entry.second->stream != nullptr) {
            checkCuda(cudaStreamSynchronize(entry.second->stream),
                      "cudaStreamSynchronize (all devices)");
        }
    }
}

DeviceScope::DeviceScope(int device) : previous_device_(current_device()) {
    if (device != previous_device_)
        set_device(device);
}

DeviceScope::~DeviceScope() {
    if (previous_device_ != current_device()) {
        cudaSetDevice(previous_device_);
    }
}

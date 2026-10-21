#include "matrix_pro/core/backend_registry.hpp"
#include "matrix_pro/core/device.hpp"
#include "matrix_pro/detail/cuda_hooks.hpp"
#include "matrix_pro/distributed/topology.hpp"

#include <string>

namespace matrix_pro {

std::unique_ptr<AcceleratorBackend> make_cpu_backend();
std::unique_ptr<AcceleratorBackend> make_rocm_backend();
std::unique_ptr<AcceleratorBackend> make_cuda_backend();

const char* device_type_name(DeviceType type) {
    switch (type) {
        case DeviceType::Cpu: return "CPU";
        case DeviceType::Cuda: return "CUDA";
        case DeviceType::Rocm: return "ROCM";
        case DeviceType::Tpu: return "TPU";
        case DeviceType::Npu: return "NPU";
        case DeviceType::Vulkan: return "VULKAN";
        case DeviceType::DirectMl: return "DIRECTML";
        case DeviceType::Metal: return "METAL";
    }
    return "UNKNOWN";
}

std::string device_label(DeviceId id) {
    return std::string(device_type_name(id.type)) + ":" + std::to_string(id.index);
}

DeviceManager& DeviceManager::instance() {
    static DeviceManager manager;
    return manager;
}

DeviceManager::DeviceManager() {
    isa_ = detect_cpu_isa();
    const MachineTopology topology = discover_topology();
    isa_.numa_nodes = topology.numa_nodes;
    isa_.logical_cores = topology.logical_cpus;
    BackendRegistry& registry = BackendRegistry::instance();
    registry.add(make_cpu_backend());
    registry.add(make_rocm_backend());
    registry.add(make_cuda_backend());
    for (AcceleratorBackend* backend : registry.all()) {
        const int count = backend->device_count();
        for (int index = 0; index < count; ++index) {
            Device device;
            device.id = DeviceId{backend->type(), index};
            device.name = backend->device_name(index);
            device.memory_bytes = backend->memory_bytes(index);
            device.available = true;
            if (backend->type() == DeviceType::Cpu) {
                device.peak_gflops_f32 = isa_.avx512f ? 80.0 : isa_.avx2 ? 40.0 : 4.0;
            }
            devices_.push_back(device);
        }
    }
}

const std::vector<Device>& DeviceManager::devices() const { return devices_; }

std::vector<Device> DeviceManager::devices_of(DeviceType type) const {
    std::vector<Device> selected;
    for (const Device& device : devices_) {
        if (device.id.type == type) {
            selected.push_back(device);
        }
    }
    return selected;
}

const CpuIsa& DeviceManager::cpu_isa() const { return isa_; }

int DeviceManager::count(DeviceType type) const { return static_cast<int>(devices_of(type).size()); }

}  // namespace matrix_pro

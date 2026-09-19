#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace matrix_pro {

class Matrix; // forward decl

struct DeviceInfo {
    int id = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t total_memory = 0;
    std::size_t free_memory = 0;
    int sm_count = 0;
    int max_threads_per_sm = 0;
    int warp_size = 32;
    int max_shared_memory_per_block = 0;
    int max_threads_per_block = 0;
    int clock_rate_khz = 0;
    int memory_clock_khz = 0;
    int memory_bus_width = 0;
    double peak_bandwidth_gbps() const;
    double peak_tflops_f32() const;
    bool supports_tf32() const;
    bool supports_fp16() const;
    bool supports_bf16() const;
    bool supports_peer_access(int other_device) const;
    std::string summary() const;
};

DeviceInfo get_device_info(int device = -1);
std::vector<DeviceInfo> get_all_devices();

// Multi-GPU primitives
void enable_peer_access(int device_a, int device_b);
Matrix copy_to_device(const Matrix& src, int target_device);
void peer_copy_async(float* dst, int dst_device, const float* src, int src_device, std::size_t count);

}

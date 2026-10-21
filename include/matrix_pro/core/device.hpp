#pragma once

/**
 * @file device.hpp
 * @brief CPU, CUDA, ROCm, TPU ve eklenti hizlandiricilarinin ortak kimligi.
 *
 * Bu baslik hesap yapmaz. Cihaz sayisi, ISA bayraklari ve adlandirma burada
 * durur. GEMM ve kolektifler kendi dosyalarindadir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace matrix_pro {

enum class DeviceType {
    Cpu = 0,
    Cuda = 1,
    Rocm = 2,
    Tpu = 3,
    Npu = 4,
    Vulkan = 5,
    DirectMl = 6,
    Metal = 7
};

struct DeviceId {
    DeviceType type = DeviceType::Cpu;
    int index = 0;

    static DeviceId cpu() { return DeviceId{DeviceType::Cpu, 0}; }
    static DeviceId cuda(int device_index) { return DeviceId{DeviceType::Cuda, device_index}; }
    static DeviceId rocm(int device_index) { return DeviceId{DeviceType::Rocm, device_index}; }
    static DeviceId tpu(int device_index) { return DeviceId{DeviceType::Tpu, device_index}; }
    static DeviceId npu(int device_index) { return DeviceId{DeviceType::Npu, device_index}; }
    static DeviceId vulkan(int device_index) { return DeviceId{DeviceType::Vulkan, device_index}; }
    static DeviceId directml(int device_index) { return DeviceId{DeviceType::DirectMl, device_index}; }
    static DeviceId metal(int device_index) { return DeviceId{DeviceType::Metal, device_index}; }

    bool operator==(DeviceId other) const { return type == other.type && index == other.index; }
    bool operator!=(DeviceId other) const { return !(*this == other); }
};

/// Calisan islemcinin komut seti. Cekirdekler yalnizca true olan yolu cagirir.
struct CpuIsa {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool amx_bf16 = false;
    bool amx_int8 = false;
    bool neon = false;
    int logical_cores = 1;
    int numa_nodes = 1;
};

struct Device {
    DeviceId id{};
    std::string name;
    std::uint64_t memory_bytes = 0;
    /// Olculmus ya da kuramsal tek-hassasiyet tepesi. 0 ise henuz olculmedi.
    double peak_gflops_f32 = 0;
    bool available = false;
};

[[nodiscard]] const char* device_type_name(DeviceType type);
[[nodiscard]] std::string device_label(DeviceId id);

[[nodiscard]] CpuIsa detect_cpu_isa();

class DeviceManager {
public:
    static DeviceManager& instance();

    [[nodiscard]] const std::vector<Device>& devices() const;
    [[nodiscard]] std::vector<Device> devices_of(DeviceType type) const;
    [[nodiscard]] const CpuIsa& cpu_isa() const;
    [[nodiscard]] int count(DeviceType type) const;

private:
    DeviceManager();
    std::vector<Device> devices_;
    CpuIsa isa_{};
};

}  // namespace matrix_pro

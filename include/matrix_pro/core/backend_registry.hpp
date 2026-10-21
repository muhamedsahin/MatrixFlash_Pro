#pragma once

/**
 * @file backend_registry.hpp
 * @brief Hizlandirici surucu kaydi ve DLL eklenti ABI'si.
 *
 * CPU ve CUDA somut suruculerdir. ROCm, hipBLAS DLL'i yuklenirse ayni arayuzu
 * doldurur. NPU, Vulkan, DirectML ve Metal bir DLL'in
 * `CreateMatrixProBackend` sembolu ile kaydolur. Eksik kutuphane sessizce
 * sifir sonuc uretmez; cihaz sayisi 0 kalir, zorlanan cagri hata firlatir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/device.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace matrix_pro {

class AcceleratorBackend {
public:
    virtual ~AcceleratorBackend() = default;
    virtual DeviceType type() const = 0;
    virtual const char* name() const = 0;
    virtual int device_count() const = 0;
    virtual std::string device_name(int index) const = 0;
    virtual std::uint64_t memory_bytes(int index) const = 0;
    virtual void* allocate(int index, std::size_t bytes) = 0;
    virtual void deallocate(int index, void* pointer) = 0;
    virtual void copy(int dst_device, void* dst, int src_device, const void* src, std::size_t bytes) = 0;
    /// Satir-major C = alpha * A * B + beta * C. A[m,k], B[k,n], C[m,n].
    virtual void gemm(int index, int m, int n, int k, const float* a, int lda, const float* b, int ldb,
                      float* c, int ldc, float alpha, float beta) = 0;
};

class BackendRegistry {
public:
    static BackendRegistry& instance();

    void add(std::unique_ptr<AcceleratorBackend> backend);
    [[nodiscard]] AcceleratorBackend* find(DeviceType type) const;
    [[nodiscard]] const std::vector<AcceleratorBackend*>& all() const;

    /// `CreateMatrixProBackend` sembolu veren bir DLL yukler. Basarisiz olursa hata firlatir.
    void load_plugin(const std::string& path);

private:
    BackendRegistry() = default;
    std::vector<std::unique_ptr<AcceleratorBackend>> owned_;
    std::vector<AcceleratorBackend*> view_;
};

/// Eklenti ABI surumu. Yapı bellegi duzeni degisirse bu sayi artar.
inline constexpr std::uint32_t kPluginAbiVersion = 1;

struct PluginGemmArgs {
    int device = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    const float* a = nullptr;
    int lda = 0;
    const float* b = nullptr;
    int ldb = 0;
    float* c = nullptr;
    int ldc = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
};

/// DLL'nin doldurdugu fonksiyon tablosu. `user` eklentinin kendi durumudur.
struct MatrixProPluginV1 {
    std::uint32_t abi = kPluginAbiVersion;
    int device_type = 0;
    const char* name = nullptr;
    void* user = nullptr;
    void (*destroy)(void* user) = nullptr;
    int (*device_count)(void* user) = nullptr;
    const char* (*device_name)(void* user, int index) = nullptr;
    std::uint64_t (*memory_bytes)(void* user, int index) = nullptr;
    void* (*allocate)(void* user, int index, std::size_t bytes) = nullptr;
    void (*deallocate)(void* user, int index, void* pointer) = nullptr;
    void (*copy)(void* user, int dst_device, void* dst, int src_device, const void* src, std::size_t bytes) = nullptr;
    void (*gemm)(void* user, const PluginGemmArgs* args) = nullptr;
};

using CreateMatrixProBackendFn = MatrixProPluginV1* (*)();

}  // namespace matrix_pro

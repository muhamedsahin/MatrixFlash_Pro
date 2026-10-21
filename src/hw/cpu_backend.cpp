#include "matrix_pro/core/backend_registry.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/cpu/gemm.hpp"

#include <cstring>
#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace matrix_pro {
namespace {

void* aligned_alloc_64(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
#if defined(_WIN32)
    void* pointer = _aligned_malloc(bytes, 64);
#else
    void* pointer = std::aligned_alloc(64, (bytes + 63u) & ~std::size_t{63});
#endif
    if (pointer == nullptr) {
        throw MatrixProError("CPU backend: aligned allocation failed");
    }
    return pointer;
}

void aligned_free_64(void* pointer) {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

class CpuBackend final : public AcceleratorBackend {
public:
    DeviceType type() const override { return DeviceType::Cpu; }
    const char* name() const override { return "cpu-avx"; }
    int device_count() const override { return 1; }
    std::string device_name(int index) const override {
        if (index != 0) {
            throw MatrixProError("CPU backend: device index out of range");
        }
        const CpuIsa isa = detect_cpu_isa();
        if (isa.avx512f) {
            return "CPU AVX-512";
        }
        if (isa.avx2) {
            return "CPU AVX2";
        }
        return "CPU scalar";
    }
    std::uint64_t memory_bytes(int) const override { return 0; }
    void* allocate(int index, std::size_t bytes) override {
        if (index != 0) {
            throw MatrixProError("CPU backend: device index out of range");
        }
        return aligned_alloc_64(bytes);
    }
    void deallocate(int, void* pointer) override { aligned_free_64(pointer); }
    void copy(int, void* dst, int, const void* src, std::size_t bytes) override {
        if (bytes == 0) {
            return;
        }
        std::memcpy(dst, src, bytes);
    }
    void gemm(int index, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
              float alpha, float beta) override {
        if (index != 0) {
            throw MatrixProError("CPU backend: device index out of range");
        }
        gemm_f32(m, n, k, a, lda, b, ldb, c, ldc, alpha, beta);
    }
};

}  // namespace

std::unique_ptr<AcceleratorBackend> make_cpu_backend() { return std::unique_ptr<AcceleratorBackend>(new CpuBackend()); }

}  // namespace matrix_pro

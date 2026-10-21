#include "matrix_pro/core/backend_registry.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/detail/cuda_hooks.hpp"

#include <memory>
#include <string>

namespace matrix_pro {
namespace hw {

namespace {

#if defined(_MSC_VER)
#define MATRIX_PRO_NOINLINE __declspec(noinline)
#else
#define MATRIX_PRO_NOINLINE __attribute__((noinline))
#endif

MATRIX_PRO_NOINLINE void unavailable(const char* what) {
    throw MatrixProError(std::string(what) + " requires the CUDA backend");
}

// Dolayli cagri: MSVC govdeyi gorse `return` satirini ulasilamaz sayar (C4702).
using ThrowFn = void (*)(const char*);
MATRIX_PRO_NOINLINE ThrowFn throw_hook() { return &unavailable; }

}  // namespace

int cuda_device_count() { return 0; }
void cuda_gemm(int, int, int, int, const float*, int, const float*, int, float*, int, float, float) {
    unavailable("cuda_gemm");
}
void cuda_copy_h2d(int, float*, const float*, std::size_t) { unavailable("cuda_copy_h2d"); }
void cuda_copy_d2h(int, float*, const float*, std::size_t) { unavailable("cuda_copy_d2h"); }
void cuda_copy_p2p(int, float*, int, const float*, std::size_t) { unavailable("cuda_copy_p2p"); }
void cuda_add_inplace(int, float*, const float*, std::size_t) { unavailable("cuda_add"); }
void cuda_max_inplace(int, float*, const float*, std::size_t) { unavailable("cuda_max"); }
void cuda_scale_inplace(int, float*, std::size_t, float) { unavailable("cuda_scale"); }
float* cuda_alloc(int, std::size_t) {
    throw_hook()("cuda_alloc");
    return nullptr;
}
void cuda_free(int, float*) {}
void cuda_sync(int) {}
bool cuda_peer_access(int, int) { return false; }
bool cuda_nvlink(int, int) { return false; }
void cuda_enable_peer(int, int) {}
const char* cuda_device_name(int) { return "cuda-absent"; }
std::uint64_t cuda_memory_bytes(int) { return 0; }
double cuda_measure_gflops(int) { return 0; }
double cuda_measure_pcie_gbps(int) { return 0; }
bool cuda_nccl_allreduce(float**, const int*, int, std::size_t, int) { return false; }
void cuda_allreduce(float**, const int*, int, std::size_t, int, int) { unavailable("cuda_allreduce"); }

}  // namespace hw

std::unique_ptr<AcceleratorBackend> make_cuda_backend() {
    class Absent final : public AcceleratorBackend {
    public:
        DeviceType type() const override { return DeviceType::Cuda; }
        const char* name() const override { return "cuda-absent"; }
        int device_count() const override { return 0; }
        std::string device_name(int) const override { return "cuda-absent"; }
        std::uint64_t memory_bytes(int) const override { return 0; }
        void* allocate(int, std::size_t) override { throw MatrixProError("CUDA backend is not compiled"); }
        void deallocate(int, void*) override {}
        void copy(int, void*, int, const void*, std::size_t) override {
            throw MatrixProError("CUDA backend is not compiled");
        }
        void gemm(int, int, int, int, const float*, int, const float*, int, float*, int, float, float) override {
            throw MatrixProError("CUDA backend is not compiled");
        }
    };
    return std::unique_ptr<AcceleratorBackend>(new Absent());
}

}  // namespace matrix_pro

#include "matrix_pro/core/backend_registry.hpp"
#include "matrix_pro/core/errors.hpp"

#include <mutex>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace matrix_pro {
namespace {

using HipGetDeviceCountFn = int (*)(int*);
using HipMallocFn = int (*)(void**, std::size_t);
using HipFreeFn = int (*)(void*);
using HipMemcpyFn = int (*)(void*, const void*, std::size_t, int);
using HipblasCreateFn = int (*)(void**);
using HipblasDestroyFn = int (*)(void*);
using HipblasSgemmFn = int (*)(void*, int, int, int, int, int, const float*, const float*, int, const float*, int,
                               const float*, float*, int);

struct RocmApi {
    HipGetDeviceCountFn get_count = nullptr;
    HipMallocFn malloc_fn = nullptr;
    HipFreeFn free_fn = nullptr;
    HipMemcpyFn memcpy_fn = nullptr;
    HipblasCreateFn create = nullptr;
    HipblasDestroyFn destroy = nullptr;
    HipblasSgemmFn sgemm = nullptr;
    void* hip_module = nullptr;
    void* blas_module = nullptr;
    bool ready = false;
};

void* load_module(const char* name) {
#if defined(_WIN32)
    return static_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_LAZY);
#endif
}

void* symbol(void* module, const char* name) {
    if (module == nullptr) {
        return nullptr;
    }
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
#else
    return dlsym(module, name);
#endif
}

RocmApi& api() {
    static RocmApi loaded = [] {
        RocmApi out;
        const char* hip_names[] = {"amdhip64.dll", "libamdhip64.so", "libamdhip64.so.5"};
        const char* blas_names[] = {"hipblas.dll", "libhipblas.so", "libhipblas.so.2"};
        for (const char* name : hip_names) {
            out.hip_module = load_module(name);
            if (out.hip_module != nullptr) {
                break;
            }
        }
        for (const char* name : blas_names) {
            out.blas_module = load_module(name);
            if (out.blas_module != nullptr) {
                break;
            }
        }
        out.get_count = reinterpret_cast<HipGetDeviceCountFn>(symbol(out.hip_module, "hipGetDeviceCount"));
        out.malloc_fn = reinterpret_cast<HipMallocFn>(symbol(out.hip_module, "hipMalloc"));
        out.free_fn = reinterpret_cast<HipFreeFn>(symbol(out.hip_module, "hipFree"));
        out.memcpy_fn = reinterpret_cast<HipMemcpyFn>(symbol(out.hip_module, "hipMemcpy"));
        out.create = reinterpret_cast<HipblasCreateFn>(symbol(out.blas_module, "hipblasCreate"));
        out.destroy = reinterpret_cast<HipblasDestroyFn>(symbol(out.blas_module, "hipblasDestroy"));
        out.sgemm = reinterpret_cast<HipblasSgemmFn>(symbol(out.blas_module, "hipblasSgemm"));
        out.ready = out.get_count && out.malloc_fn && out.free_fn && out.memcpy_fn && out.create && out.sgemm;
        return out;
    }();
    return loaded;
}

class RocmBackend final : public AcceleratorBackend {
public:
    DeviceType type() const override { return DeviceType::Rocm; }
    const char* name() const override { return "rocm-hipblas"; }
    int device_count() const override {
        if (!api().ready) {
            return 0;
        }
        int count = 0;
        if (api().get_count(&count) != 0 || count < 0) {
            return 0;
        }
        return count;
    }
    std::string device_name(int index) const override { return "ROCM:" + std::to_string(index); }
    std::uint64_t memory_bytes(int) const override { return 0; }
    void* allocate(int, std::size_t bytes) override {
        require();
        void* pointer = nullptr;
        if (api().malloc_fn(&pointer, bytes) != 0 || pointer == nullptr) {
            throw MatrixProError("ROCm hipMalloc failed");
        }
        return pointer;
    }
    void deallocate(int, void* pointer) override {
        if (pointer != nullptr && api().ready) {
            api().free_fn(pointer);
        }
    }
    void copy(int, void* dst, int, const void* src, std::size_t bytes) override {
        require();
        // hipMemcpyDeviceToDevice = 3. Host staging is the caller's duty via gemm.
        if (api().memcpy_fn(dst, src, bytes, 3) != 0) {
            throw MatrixProError("ROCm hipMemcpy failed");
        }
    }
    void gemm(int, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
              float alpha, float beta) override {
        require();
        void* handle = nullptr;
        if (api().create(&handle) != 0 || handle == nullptr) {
            throw MatrixProError("ROCm hipblasCreate failed");
        }
        // hipBLAS OP_N is 111. Row-major C=A*B is column-major C^T = B^T * A^T,
        // passed as N,N with B then A and n,m swapped.
        constexpr int kOpN = 111;
        const int status = api().sgemm(handle, kOpN, kOpN, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc);
        if (api().destroy != nullptr) {
            api().destroy(handle);
        }
        if (status != 0) {
            throw MatrixProError("ROCm hipblasSgemm failed");
        }
    }

private:
    static void require() {
        if (!api().ready) {
            throw MatrixProError("ROCm runtime is not installed");
        }
    }
};

class PluginBackend final : public AcceleratorBackend {
public:
    explicit PluginBackend(MatrixProPluginV1 table) : table_(table) {
        if (table_.abi != kPluginAbiVersion || table_.gemm == nullptr || table_.device_count == nullptr) {
            throw MatrixProError("plugin ABI is not version 1");
        }
    }
    ~PluginBackend() override {
        if (table_.destroy != nullptr) {
            table_.destroy(table_.user);
        }
    }
    DeviceType type() const override { return static_cast<DeviceType>(table_.device_type); }
    const char* name() const override { return table_.name != nullptr ? table_.name : "plugin"; }
    int device_count() const override { return table_.device_count(table_.user); }
    std::string device_name(int index) const override {
        if (table_.device_name == nullptr) {
            return name();
        }
        const char* label = table_.device_name(table_.user, index);
        return label != nullptr ? std::string(label) : name();
    }
    std::uint64_t memory_bytes(int index) const override {
        return table_.memory_bytes != nullptr ? table_.memory_bytes(table_.user, index) : 0;
    }
    void* allocate(int index, std::size_t bytes) override {
        if (table_.allocate == nullptr) {
            throw MatrixProError("plugin has no allocator");
        }
        return table_.allocate(table_.user, index, bytes);
    }
    void deallocate(int index, void* pointer) override {
        if (table_.deallocate != nullptr) {
            table_.deallocate(table_.user, index, pointer);
        }
    }
    void copy(int dst_device, void* dst, int src_device, const void* src, std::size_t bytes) override {
        if (table_.copy == nullptr) {
            throw MatrixProError("plugin has no copy");
        }
        table_.copy(table_.user, dst_device, dst, src_device, src, bytes);
    }
    void gemm(int index, int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
              float alpha, float beta) override {
        PluginGemmArgs args;
        args.device = index;
        args.m = m;
        args.n = n;
        args.k = k;
        args.a = a;
        args.lda = lda;
        args.b = b;
        args.ldb = ldb;
        args.c = c;
        args.ldc = ldc;
        args.alpha = alpha;
        args.beta = beta;
        table_.gemm(table_.user, &args);
    }

private:
    MatrixProPluginV1 table_;
};

std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace

std::unique_ptr<AcceleratorBackend> make_rocm_backend() {
    return std::unique_ptr<AcceleratorBackend>(new RocmBackend());
}

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    return registry;
}

void BackendRegistry::add(std::unique_ptr<AcceleratorBackend> backend) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    view_.push_back(backend.get());
    owned_.push_back(std::move(backend));
}

AcceleratorBackend* BackendRegistry::find(DeviceType type) const {
    for (AcceleratorBackend* backend : view_) {
        if (backend->type() == type) {
            return backend;
        }
    }
    return nullptr;
}

const std::vector<AcceleratorBackend*>& BackendRegistry::all() const { return view_; }

void BackendRegistry::load_plugin(const std::string& path) {
    void* module = load_module(path.c_str());
    if (module == nullptr) {
        throw MatrixProError("plugin could not be loaded: " + path);
    }
    auto* create = reinterpret_cast<CreateMatrixProBackendFn>(symbol(module, "CreateMatrixProBackend"));
    if (create == nullptr) {
        throw MatrixProError("plugin is missing CreateMatrixProBackend");
    }
    MatrixProPluginV1* table = create();
    if (table == nullptr) {
        throw MatrixProError("plugin returned a null backend");
    }
    add(std::unique_ptr<AcceleratorBackend>(new PluginBackend(*table)));
}

}  // namespace matrix_pro

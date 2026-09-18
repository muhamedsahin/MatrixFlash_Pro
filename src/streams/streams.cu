#include "matrix_pro/streams/stream_pool.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace matrix_pro {
namespace {

struct PoolState {
    std::mutex mutex;
    std::unordered_map<int, std::vector<cudaStream_t>> streams;
    ~PoolState() {
        for (auto& dev : streams)
            for (cudaStream_t s : dev.second)
                if (s) cudaStreamDestroy(s);
    }
};

PoolState& pool_state() {
    static PoolState state;
    return state;
}

__global__ void arg_shuffle_kernel(const float* in, std::size_t n,
                                   float* best_val, std::size_t* best_idx, int find_max) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float best = find_max ? -3.402823466e+38f : 3.402823466e+38f;
    std::size_t best_i = tid;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(in) & 0xFu) == 0u;
    if (aligned && n >= 4) {
        const std::size_t n4 = n / 4;
        const auto* v4 = reinterpret_cast<const float4*>(in);
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = tid; i < n4; i += stride) {
            const float4 v = v4[i];
            const float c[4] = {v.x, v.y, v.z, v.w};
            for (int k = 0; k < 4; ++k) {
                const std::size_t idx = i * 4 + static_cast<std::size_t>(k);
                if ((find_max && c[k] > best) || (!find_max && c[k] < best)) {
                    best = c[k]; best_i = idx;
                }
            }
        }
        for (std::size_t i = n4 * 4 + tid; i < n; i += stride) {
            const float v = in[i];
            if ((find_max && v > best) || (!find_max && v < best)) { best = v; best_i = i; }
        }
    } else {
        const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t i = tid; i < n; i += stride) {
            const float v = in[i];
            if ((find_max && v > best) || (!find_max && v < best)) { best = v; best_i = i; }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, offset);
        const unsigned oi = __shfl_down_sync(0xffffffffu, static_cast<unsigned>(best_i), offset);
        if ((find_max && ov > best) || (!find_max && ov < best) ||
            (ov == best && oi < static_cast<unsigned>(best_i))) {
            best = ov; best_i = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        const unsigned slot = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        best_val[slot] = best;
        best_idx[slot] = best_i;
    }
}

struct AsyncArgTask {
    float* host_vals = nullptr;
    std::size_t* host_idx = nullptr;
    float* dev_vals = nullptr;
    std::size_t* dev_idx = nullptr;
    std::size_t slots = 0;
    std::size_t* out = nullptr;
    std::function<void(std::size_t)> cb;
    int find_max = 1;
};

void CUDART_CB async_arg_callback(cudaStream_t stream, cudaError_t status, void* user) {
    (void)stream;
    std::unique_ptr<AsyncArgTask> task(static_cast<AsyncArgTask*>(user));
    if (status != cudaSuccess) return;
    float best = task->find_max ? -3.402823466e+38f : 3.402823466e+38f;
    std::size_t best_i = 0;
    for (std::size_t i = 0; i < task->slots; ++i) {
        const float v = task->host_vals[i];
        const std::size_t idx = task->host_idx[i];
        if (i == 0 || (task->find_max ? (v > best) : (v < best)) ||
            (v == best && idx < best_i)) { best = v; best_i = idx; }
    }
    *task->out = best_i;
    if (task->cb) task->cb(best_i);
    cudaFreeHost(task->host_vals);
    cudaFreeHost(task->host_idx);
    cudaFree(task->dev_vals);
    cudaFree(task->dev_idx);
}

void launch_arg_async(const Matrix& m, std::size_t* out,
                      const std::function<void(std::size_t)>& cb, int find_max) {
    if (m.empty()) throw InvalidArgumentError("argmax/argmin of empty matrix");
    if (out == nullptr) throw InvalidArgumentError("async out pointer is null");
    const std::size_t n = m.size();
    unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    if (blocks > 128) blocks = 128;
    if (blocks == 0) blocks = 1;
    const std::size_t slots = static_cast<std::size_t>(blocks) * 8;
    float* dev_vals = nullptr;
    std::size_t* dev_idx = nullptr;
    float* host_vals = nullptr;
    std::size_t* host_idx = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev_vals), slots * sizeof(float)), "async alloc");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dev_idx), slots * sizeof(std::size_t)), "async alloc");
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&host_vals), slots * sizeof(float),
                            cudaHostAllocDefault), "async pin");
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&host_idx), slots * sizeof(std::size_t),
                            cudaHostAllocDefault), "async pin");
    arg_shuffle_kernel<<<blocks, 256, 0, compute_stream()>>>(
        m.device_data(), n, dev_vals, dev_idx, find_max);
    checkCuda(cudaGetLastError(), "arg shuffle kernel launch");
    checkCuda(cudaMemcpyAsync(host_vals, dev_vals, slots * sizeof(float),
                              cudaMemcpyDeviceToHost, compute_stream()), "async read");
    checkCuda(cudaMemcpyAsync(host_idx, dev_idx, slots * sizeof(std::size_t),
                              cudaMemcpyDeviceToHost, compute_stream()), "async read");
    auto* task = new AsyncArgTask{host_vals, host_idx, dev_vals, dev_idx, slots, out, cb, find_max};
    checkCuda(cudaStreamAddCallback(compute_stream(), async_arg_callback, task, 0),
              "cudaStreamAddCallback");
}

} // namespace


cudaStream_t pool_stream(int slot) {
    int device = 0;
    cudaGetDevice(&device);
    auto& state = pool_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto& vec = state.streams[device];
    const auto idx = static_cast<std::size_t>(slot < 0 ? 0 : slot) % kStreamPoolSize;
    while (vec.size() <= idx) {
        cudaStream_t s = nullptr;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess)
            throw CudaError("pool stream creation failed");
        vec.push_back(s);
    }
    return vec[idx];
}

void synchronize_pool() {
    int device = 0;
    cudaGetDevice(&device);
    auto& state = pool_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.streams.find(device);
    if (it == state.streams.end()) return;
    for (cudaStream_t s : it->second)
        if (s) checkCuda(cudaStreamSynchronize(s), "pool stream sync");
}

void launch_on_pool(int slot, const std::function<void(cudaStream_t)>& fn, cudaEvent_t done) {
    cudaStream_t s = pool_stream(slot);
    fn(s);
    if (done) checkCuda(cudaEventRecord(done, s), "pool event record");
}

cudaEvent_t make_event() {
    cudaEvent_t e = nullptr;
    checkCuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event create");
    return e;
}

void destroy_event(cudaEvent_t event) {
    if (event) cudaEventDestroy(event);
}

void* pin_host(std::size_t bytes) {
    void* p = nullptr;
    checkCuda(cudaHostAlloc(&p, bytes, cudaHostAllocDefault), "pin_host");
    return p;
}

void unpin_host(void* ptr) {
    if (ptr) checkCuda(cudaFreeHost(ptr), "unpin_host");
}

void argmax_async(const Matrix& matrix, std::size_t* out,
                  const std::function<void(std::size_t)>& callback) {
    launch_arg_async(matrix, out, callback, 1);
}

void argmin_async(const Matrix& matrix, std::size_t* out,
                  const std::function<void(std::size_t)>& callback) {
    launch_arg_async(matrix, out, callback, 0);
}

} // namespace matrix_pro



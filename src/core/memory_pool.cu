#include "matrix_pro/core/memory_pool.hpp"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <mutex>

namespace matrix_pro {

inline void checkCudaMem(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA Error in memory pool: " << msg << " - " << cudaGetErrorString(result);
        throw std::runtime_error(ss.str());
    }
}

static MemoryStats g_stats;
static std::size_t g_memory_limit = static_cast<std::size_t>(-1);
static std::mutex g_pool_mutex;

double MemoryStats::cache_hit_rate() const {
    if (num_allocations == 0) return 0.0;
    return static_cast<double>(num_cache_hits) / static_cast<double>(num_allocations);
}

std::string MemoryStats::summary() const {
    std::stringstream ss;
    ss << "Memory Pool Stats:\n";
    ss << "  Current Allocated: " << (current_allocated / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Peak Allocated: " << (peak_allocated / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Total Allocated: " << (total_allocated / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Total Freed: " << (total_freed / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Pool Cached: " << (pool_cached / (1024.0 * 1024.0)) << " MB\n";
    ss << "  Number of Allocations: " << num_allocations << "\n";
    ss << "  Cache Hits: " << num_cache_hits << "\n";
    ss << "  Cache Hit Rate: " << std::fixed << std::setprecision(2) << (cache_hit_rate() * 100.0) << "%\n";
    return ss.str();
}

MemoryStats memory_stats() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    return g_stats;
}

void reset_peak_memory() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_stats.peak_allocated = g_stats.current_allocated;
}

void empty_cache() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    // Placeholder: empty internal device pool structures
    g_stats.pool_cached = 0;
}

void set_memory_limit(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_memory_limit = bytes;
}

std::size_t memory_limit() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    return g_memory_limit;
}

void* allocate_unified_memory(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_stats.current_allocated + bytes > g_memory_limit) {
        throw std::runtime_error("Memory limit exceeded");
    }
    
    void* ptr = nullptr;
    checkCudaMem(cudaMallocManaged(&ptr, bytes), "cudaMallocManaged");
    
    g_stats.current_allocated += bytes;
    g_stats.total_allocated += bytes;
    if (g_stats.current_allocated > g_stats.peak_allocated) {
        g_stats.peak_allocated = g_stats.current_allocated;
    }
    g_stats.num_allocations++;
    
    return ptr;
}

void free_unified_memory(void* ptr) {
    if (!ptr) return;
    checkCudaMem(cudaFree(ptr), "cudaFree");
    // Size is not tracked per pointer in this simple wrapper, so stats update would require a map.
    // Assuming this is handled if a map is used, but for simplicity here we just free.
}

void prefetch_to_device(void* ptr, std::size_t bytes, int device) {
    if (!ptr || bytes == 0) return;
    
    if (device == -1) {
        checkCudaMem(cudaGetDevice(&device), "cudaGetDevice");
    }
    
    checkCudaMem(cudaMemPrefetchAsync(ptr, bytes, device, nullptr), "cudaMemPrefetchAsync");
}

void prefetch_to_host(void* ptr, std::size_t bytes) {
    if (!ptr || bytes == 0) return;
    
    checkCudaMem(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, nullptr), "cudaMemPrefetchAsync");
}

}

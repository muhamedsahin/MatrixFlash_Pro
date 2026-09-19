#pragma once
#include <cstddef>
#include <string>

namespace matrix_pro {

struct MemoryStats {
    std::size_t current_allocated = 0;
    std::size_t peak_allocated = 0;
    std::size_t total_allocated = 0;
    std::size_t total_freed = 0;
    std::size_t pool_cached = 0;
    std::size_t num_allocations = 0;
    std::size_t num_cache_hits = 0;
    double cache_hit_rate() const;
    std::string summary() const;
};

MemoryStats memory_stats();
void reset_peak_memory();
void empty_cache();
void set_memory_limit(std::size_t bytes);
std::size_t memory_limit();

// Unified memory (CPU+GPU automatic migration)
void* allocate_unified_memory(std::size_t bytes);
void free_unified_memory(void* ptr);
void prefetch_to_device(void* ptr, std::size_t bytes, int device = -1);
void prefetch_to_host(void* ptr, std::size_t bytes);

}

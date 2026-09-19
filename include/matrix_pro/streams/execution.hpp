#pragma once
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace matrix_pro {

// CUDA Graph capture & replay — eliminates kernel launch overhead
class CudaGraph {
public:
    CudaGraph() = default;
    ~CudaGraph();
    CudaGraph(const CudaGraph&) = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&& other) noexcept;
    CudaGraph& operator=(CudaGraph&& other) noexcept;

    void begin_capture();
    void end_capture();
    void replay();
    bool is_capturing() const noexcept;
    bool is_compiled() const noexcept;
    void reset();

private:
    void* graph_ = nullptr;
    void* exec_ = nullptr;
    bool capturing_ = false;
};

// Multi-stream pipeline for overlapping compute+transfer
class Pipeline {
public:
    explicit Pipeline(int num_stages = 4);
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void enqueue(std::function<void()> fn);
    void barrier();
    void synchronize();
    int num_stages() const noexcept;
    int current_stage() const noexcept;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// Kernel execution profiler
struct KernelTiming {
    std::string label;
    float elapsed_ms = 0.0f;
};

class ExecutionProfiler {
public:
    ExecutionProfiler() = default;
    ~ExecutionProfiler();

    void begin(const std::string& label = "");
    void end();
    float last_elapsed_ms() const;
    std::vector<KernelTiming> results() const;
    float total_ms() const;
    void reset();
    std::string summary() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}

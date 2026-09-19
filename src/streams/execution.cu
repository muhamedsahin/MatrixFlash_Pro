#include "matrix_pro/streams/execution.hpp"
#include "matrix_pro/utils/cuda_utils.hpp"
#include "matrix_pro/core/errors.hpp"
#include <cuda_runtime.h>
#include <sstream>

namespace matrix_pro {

// CudaGraph Implementation
CudaGraph::~CudaGraph() {
    reset();
}

CudaGraph::CudaGraph(CudaGraph&& other) noexcept 
    : graph_(other.graph_), exec_(other.exec_), capturing_(other.capturing_) {
    other.graph_ = nullptr;
    other.exec_ = nullptr;
    other.capturing_ = false;
}

CudaGraph& CudaGraph::operator=(CudaGraph&& other) noexcept {
    if (this != &other) {
        reset();
        graph_ = other.graph_;
        exec_ = other.exec_;
        capturing_ = other.capturing_;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
        other.capturing_ = false;
    }
    return *this;
}

void CudaGraph::begin_capture() {
    if (capturing_) throw InvalidArgumentError("Graph is already capturing.");
    reset();
    cudaStream_t stream = detail::compute_stream();
    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    capturing_ = true;
}

void CudaGraph::end_capture() {
    if (!capturing_) throw InvalidArgumentError("Graph is not capturing.");
    cudaStream_t stream = detail::compute_stream();
    cudaGraph_t graph_temp;
    checkCuda(cudaStreamEndCapture(stream, &graph_temp), "cudaStreamEndCapture");
    
    cudaGraphExec_t exec_temp;
    // Note: CUDA 11+ instantiation. nullptr params for error logging are omitted for brevity per standard use cases.
    checkCuda(cudaGraphInstantiate(&exec_temp, graph_temp, nullptr, nullptr, 0), "cudaGraphInstantiate");
    
    checkCuda(cudaGraphDestroy(graph_temp), "cudaGraphDestroy");
    exec_ = exec_temp;
    capturing_ = false;
}

void CudaGraph::replay() {
    if (!is_compiled()) throw InvalidArgumentError("Graph is not compiled.");
    checkCuda(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_), detail::compute_stream()), "cudaGraphLaunch");
}

bool CudaGraph::is_capturing() const noexcept {
    return capturing_;
}

bool CudaGraph::is_compiled() const noexcept {
    return exec_ != nullptr;
}

void CudaGraph::reset() {
    if (exec_ != nullptr) {
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_));
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
        graph_ = nullptr;
    }
    capturing_ = false;
}


// Pipeline Implementation
struct Pipeline::Impl {
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    int num_stages;
    int current_stage = 0;
};

Pipeline::Pipeline(int num_stages) {
    if (num_stages <= 0) throw InvalidArgumentError("Pipeline must have at least 1 stage.");
    impl_ = new Impl();
    impl_->num_stages = num_stages;
    impl_->streams.resize(num_stages);
    impl_->events.resize(num_stages);
    for (int i = 0; i < num_stages; ++i) {
        checkCuda(cudaStreamCreateWithFlags(&impl_->streams[i], cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        checkCuda(cudaEventCreateWithFlags(&impl_->events[i], cudaEventDisableTiming), "cudaEventCreateWithFlags");
    }
}

Pipeline::~Pipeline() {
    if (impl_) {
        for (int i = 0; i < impl_->num_stages; ++i) {
            cudaStreamDestroy(impl_->streams[i]);
            cudaEventDestroy(impl_->events[i]);
        }
        delete impl_;
    }
}

namespace detail {
    // Expected forward declaration for stream guard / setter.
    extern void set_compute_stream(cudaStream_t stream);
}

void Pipeline::enqueue(std::function<void()> fn) {
    int stage = impl_->current_stage;
    cudaStream_t stream = impl_->streams[stage];
    
    cudaStream_t old_stream = detail::compute_stream();
    detail::set_compute_stream(stream);
    
    fn();
    
    checkCuda(cudaEventRecord(impl_->events[stage], stream), "cudaEventRecord");
    detail::set_compute_stream(old_stream);
    
    impl_->current_stage = (stage + 1) % impl_->num_stages;
}

void Pipeline::barrier() {
    int stage = impl_->current_stage;
    int prev_stage = (stage - 1 + impl_->num_stages) % impl_->num_stages;
    cudaStream_t stream = impl_->streams[stage];
    checkCuda(cudaStreamWaitEvent(stream, impl_->events[prev_stage], 0), "cudaStreamWaitEvent");
}

void Pipeline::synchronize() {
    for (int i = 0; i < impl_->num_stages; ++i) {
        checkCuda(cudaStreamSynchronize(impl_->streams[i]), "cudaStreamSynchronize");
    }
}

int Pipeline::num_stages() const noexcept {
    return impl_->num_stages;
}

int Pipeline::current_stage() const noexcept {
    return impl_->current_stage;
}


// ExecutionProfiler Implementation
struct ProfilerEventPair {
    cudaEvent_t start;
    cudaEvent_t stop;
    std::string label;
};

struct ExecutionProfiler::Impl {
    std::vector<ProfilerEventPair> events;
    float last_elapsed = 0.0f;
};

ExecutionProfiler::ExecutionProfiler() {
    impl_ = new Impl();
}

ExecutionProfiler::~ExecutionProfiler() {
    if (impl_) {
        reset();
        delete impl_;
    }
}

void ExecutionProfiler::begin(const std::string& label) {
    ProfilerEventPair pair;
    pair.label = label;
    checkCuda(cudaEventCreate(&pair.start), "cudaEventCreate (start)");
    checkCuda(cudaEventCreate(&pair.stop), "cudaEventCreate (stop)");
    checkCuda(cudaEventRecord(pair.start, detail::compute_stream()), "cudaEventRecord (start)");
    impl_->events.push_back(pair);
}

void ExecutionProfiler::end() {
    if (impl_->events.empty()) throw InvalidArgumentError("No active profiler event.");
    
    auto& pair = impl_->events.back();
    checkCuda(cudaEventRecord(pair.stop, detail::compute_stream()), "cudaEventRecord (stop)");
    checkCuda(cudaEventSynchronize(pair.stop), "cudaEventSynchronize");
    
    float ms = 0;
    checkCuda(cudaEventElapsedTime(&ms, pair.start, pair.stop), "cudaEventElapsedTime");
    impl_->last_elapsed = ms;
}

float ExecutionProfiler::last_elapsed_ms() const {
    return impl_->last_elapsed;
}

std::vector<KernelTiming> ExecutionProfiler::results() const {
    std::vector<KernelTiming> res;
    for (const auto& pair : impl_->events) {
        float ms = 0;
        cudaEventElapsedTime(&ms, pair.start, pair.stop);
        res.push_back({pair.label, ms});
    }
    return res;
}

float ExecutionProfiler::total_ms() const {
    float total = 0.0f;
    for (const auto& pair : impl_->events) {
        float ms = 0;
        cudaEventElapsedTime(&ms, pair.start, pair.stop);
        total += ms;
    }
    return total;
}

void ExecutionProfiler::reset() {
    for (auto& pair : impl_->events) {
        cudaEventDestroy(pair.start);
        cudaEventDestroy(pair.stop);
    }
    impl_->events.clear();
    impl_->last_elapsed = 0.0f;
}

std::string ExecutionProfiler::summary() const {
    std::ostringstream oss;
    oss << "ExecutionProfiler Summary:\n";
    float total = 0.0f;
    for (const auto& pair : impl_->events) {
        float ms = 0;
        cudaEventElapsedTime(&ms, pair.start, pair.stop);
        oss << "  " << (pair.label.empty() ? "unnamed" : pair.label) << ": " << ms << " ms\n";
        total += ms;
    }
    oss << "Total: " << total << " ms\n";
    return oss.str();
}

} // namespace matrix_pro

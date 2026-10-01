// Dahili uygulama parçası: src/streams/execution.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Event tabanlı execution profiler; ölçüm ve senkronizasyon noktaları.

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
    checkCuda(cudaEventRecord(pair.start, compute_stream()), "cudaEventRecord (start)");
    impl_->events.push_back(pair);
}

void ExecutionProfiler::end() {
    if (impl_->events.empty())
        throw InvalidArgumentError("No active profiler event.");

    auto& pair = impl_->events.back();
    checkCuda(cudaEventRecord(pair.stop, compute_stream()), "cudaEventRecord (stop)");
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

// Dahili uygulama parçası: src/streams/execution.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Pipeline stream/event sıralaması ve exception sonrası stream geri yükleme.

// Pipeline Implementation
struct Pipeline::Impl {
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    int num_stages;
    int current_stage = 0;
};

Pipeline::Pipeline(int num_stages) {
    if (num_stages <= 0)
        throw InvalidArgumentError("Pipeline must have at least 1 stage.");
    impl_ = new Impl();
    impl_->num_stages = num_stages;
    impl_->streams.resize(num_stages);
    impl_->events.resize(num_stages);
    for (int i = 0; i < num_stages; ++i) {
        checkCuda(cudaStreamCreateWithFlags(&impl_->streams[i], cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
        checkCuda(cudaEventCreateWithFlags(&impl_->events[i], cudaEventDisableTiming),
                  "cudaEventCreateWithFlags");
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

    cudaStream_t old_stream = compute_stream();
    detail::set_compute_stream(stream);

    try {
        fn();
        checkCuda(cudaEventRecord(impl_->events[stage], stream), "cudaEventRecord");
    } catch (...) {
        detail::set_compute_stream(old_stream);
        throw;
    }
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

// Dahili uygulama parçası: src/streams/execution.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: CUDA Graph capture/replay/reset; yarım capture temizliği korunur.

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
    if (capturing_)
        throw InvalidArgumentError("Graph is already capturing.");
    reset();
    cudaStream_t stream = compute_stream();
    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
              "cudaStreamBeginCapture");
    capturing_ = true;
}

void CudaGraph::end_capture() {
    if (!capturing_)
        throw InvalidArgumentError("Graph is not capturing.");
    cudaStream_t stream = compute_stream();
    cudaGraph_t graph_temp = nullptr;
    const auto status = cudaStreamEndCapture(stream, &graph_temp);
    capturing_ = false;
    checkCuda(status, "cudaStreamEndCapture");

    cudaGraphExec_t exec_temp = nullptr;
    // Note: CUDA 11+ instantiation. nullptr params for error logging are omitted for brevity per
    // standard use cases.
    const auto instantiate_status =
        cudaGraphInstantiate(&exec_temp, graph_temp, nullptr, nullptr, 0);

    checkCuda(cudaGraphDestroy(graph_temp), "cudaGraphDestroy");
    checkCuda(instantiate_status, "cudaGraphInstantiate");
    exec_ = exec_temp;
    capturing_ = false;
}

void CudaGraph::replay() {
    if (!is_compiled())
        throw InvalidArgumentError("Graph is not compiled.");
    checkCuda(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_), compute_stream()),
              "cudaGraphLaunch");
}

bool CudaGraph::is_capturing() const noexcept {
    return capturing_;
}

bool CudaGraph::is_compiled() const noexcept {
    return exec_ != nullptr;
}

void CudaGraph::reset() {
    if (capturing_) {
        cudaGraph_t discarded = nullptr;
        cudaStreamEndCapture(compute_stream(), &discarded);
        if (discarded)
            cudaGraphDestroy(discarded);
        capturing_ = false;
    }
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

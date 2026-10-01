// Dahili uygulama parçası: src/rng/rng.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Bağımsız RNGState seed/counter ile mevcut tekrarlanabilirlik.

RNGState::RNGState(unsigned long long seed) : seed_(seed), counter_(0ULL) {
}

void RNGState::manual_seed(unsigned long long seed) {
    seed_ = seed;
    counter_ = 0ULL;
}

unsigned long long RNGState::seed() const noexcept {
    return seed_;
}

Matrix RNGState::randn(std::size_t rows, std::size_t cols) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    box_muller_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), seed_, counter_);
    counter_ += out.size() * 2;
    checkCuda(cudaGetLastError(), "box muller kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix RNGState::uniform(std::size_t rows, std::size_t cols, float low, float high) {
    if (high < low)
        throw InvalidArgumentError("uniform high must be >= low");
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    uniform_philox_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high - low, seed_, counter_);
    counter_ += out.size();
    checkCuda(cudaGetLastError(), "uniform kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix RNGState::bernoulli(std::size_t rows, std::size_t cols, float prob) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    bernoulli_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), prob, seed_, counter_);
    counter_ += out.size();
    checkCuda(cudaGetLastError(), "bernoulli kernel launch");
    out.mark_host_stale();
    return out;
}

// Dahili uygulama parçası: src/rng/rng.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Seed, normal, uniform ve dropout girişleri.

void rng_seed(unsigned long long seed) {
    global_seed().store(seed);
    global_counter().store(0ULL);
}

unsigned long long rng_seed_value() {
    return global_seed().load();
}

Matrix randn_gpu(std::size_t rows, std::size_t cols) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    box_muller_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), global_seed().load(), next_counter(out.size() * 2));
    checkCuda(cudaGetLastError(), "box muller kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix uniform_gpu(std::size_t rows, std::size_t cols, float low, float high) {
    if (high < low)
        throw InvalidArgumentError("uniform high must be >= low");
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    uniform_philox_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high - low, global_seed().load(),
        next_counter(out.size()));
    checkCuda(cudaGetLastError(), "uniform kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix dropout_gpu(const Matrix& matrix, float probability) {
    if (probability < 0.0f || probability >= 1.0f)
        throw InvalidArgumentError("Dropout probability must be in [0, 1)");
    if (probability == 0.0f)
        return matrix;
    Matrix out(matrix.rows(), matrix.cols());
    dropout_philox_kernel<<<(matrix.size() + 255) / 256, 256, 0, compute_stream()>>>(
        matrix.device_data(), out.device_data(), matrix.size(), probability, global_seed().load(),
        next_counter(matrix.size()));
    checkCuda(cudaGetLastError(), "dropout philox kernel launch");
    out.mark_host_stale();
    return out;
}

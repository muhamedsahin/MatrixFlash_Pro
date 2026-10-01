// Dahili uygulama parçası: src/rng/rng.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Genişletilmiş dağılımlar ve randperm host sıralama yolu.

Matrix bernoulli_gpu(std::size_t rows, std::size_t cols, float prob) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    bernoulli_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), prob, global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "bernoulli kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix exponential_gpu(std::size_t rows, std::size_t cols, float lambda) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    exponential_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), lambda, global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "exponential kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix poisson_gpu(std::size_t rows, std::size_t cols, float lambda) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    unsigned long long ctr_inc = (lambda <= 30.0f)
                                     ? out.size() * static_cast<unsigned long long>(lambda + 5.0f)
                                     : out.size() * 2;
    poisson_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), lambda, global_seed().load(), next_counter(ctr_inc));
    checkCuda(cudaGetLastError(), "poisson kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix cauchy_gpu(std::size_t rows, std::size_t cols, float loc, float scale) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    cauchy_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), loc, scale, global_seed().load(), next_counter(out.size()));
    checkCuda(cudaGetLastError(), "cauchy kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix log_normal_gpu(std::size_t rows, std::size_t cols, float mean, float stddev) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    log_normal_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), mean, stddev, global_seed().load(),
        next_counter(out.size() * 2));
    checkCuda(cudaGetLastError(), "log normal kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix truncated_normal_gpu(std::size_t rows, std::size_t cols, float low, float high) {
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    unsigned long long ctr_inc = out.size() * 10;
    truncated_normal_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high, global_seed().load(), next_counter(ctr_inc));
    checkCuda(cudaGetLastError(), "truncated normal kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix randint_gpu(std::size_t rows, std::size_t cols, int low, int high) {
    if (high <= low)
        throw InvalidArgumentError("randint high must be > low");
    Matrix out(rows, cols);
    if (out.empty())
        return out;
    randint_kernel<<<(out.size() + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), out.size(), low, high - low, global_seed().load(),
        next_counter(out.size()));
    checkCuda(cudaGetLastError(), "randint kernel launch");
    out.mark_host_stale();
    return out;
}

Matrix randperm_gpu(std::size_t n) {
    Matrix out(n, 1);
    if (n == 0)
        return out;
    Matrix keys(n, 1);
    randperm_init_kernel<<<(n + 255) / 256, 256, 0, compute_stream()>>>(
        out.device_data(), keys.device_data(), n, global_seed().load() ^ next_counter(n));
    checkCuda(cudaGetLastError(), "randperm_init_kernel launch");

    out.download();
    keys.download();
    std::vector<std::pair<float, float>> pairs(n);
    for (std::size_t i = 0; i < n; ++i) {
        pairs[i] = {keys.data()[i], out.data()[i]};
    }
    std::sort(pairs.begin(), pairs.end());
    for (std::size_t i = 0; i < n; ++i) {
        out.data()[i] = pairs[i].second;
    }
    out.upload();
    return out;
}

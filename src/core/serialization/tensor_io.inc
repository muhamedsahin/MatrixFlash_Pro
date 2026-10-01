// Dahili uygulama parçası: src/core/serialization.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Tensor dosya formatı; boyutlar ve değerler yüklemede yeniden kurulur.

void save_tensor(const Tensor& tensor, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw IoError("Failed to open file for writing: " + filename);

    uint32_t magic = MFT_MAGIC;
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    const auto& shape = tensor.shape();
    uint32_t rank = static_cast<uint32_t>(shape.size());
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

    std::vector<uint64_t> shape_u64(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        shape_u64[i] = static_cast<uint64_t>(shape[i]);
    }
    out.write(reinterpret_cast<const char*>(shape_u64.data()), rank * sizeof(uint64_t));

    size_t data_size = tensor.size() * sizeof(float);
    std::vector<float> host_data(tensor.size());

    checkCuda(cudaMemcpyAsync(host_data.data(), tensor.device_data(), data_size,
                              cudaMemcpyDeviceToHost, compute_stream()),
              "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");

    out.write(reinterpret_cast<const char*>(host_data.data()), data_size);
    if (!out.good()) {
        throw IoError("Failed to write tensor data to file: " + filename);
    }
}

Tensor load_tensor(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw IoError("Failed to open file for reading: " + filename);

    uint32_t magic = 0;
    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != MFT_MAGIC || version != 1) {
        throw InvalidArgumentError("Invalid tensor file format or version: " + filename);
    }

    uint32_t rank = 0;
    in.read(reinterpret_cast<char*>(&rank), sizeof(rank));

    std::vector<uint64_t> shape_u64(rank);
    in.read(reinterpret_cast<char*>(shape_u64.data()), rank * sizeof(uint64_t));

    std::vector<size_t> shape(rank);
    size_t total_elements = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        shape[i] = static_cast<size_t>(shape_u64[i]);
        total_elements *= shape[i];
    }

    std::vector<float> host_data(total_elements);
    size_t data_size = total_elements * sizeof(float);
    in.read(reinterpret_cast<char*>(host_data.data()), data_size);

    if (!in.good()) {
        throw IoError("Failed to read tensor data from file: " + filename);
    }

    Tensor tensor(shape);
    checkCuda(cudaMemcpyAsync(tensor.device_data(), host_data.data(), data_size,
                              cudaMemcpyHostToDevice, compute_stream()),
              "cudaMemcpyAsync H2D");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");

    tensor.mark_host_stale();
    return tensor;
}

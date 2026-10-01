// Dahili uygulama parçası: src/core/serialization.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Anahtar sorguları, Matrix/Tensor erişimi ve metadata.

Matrix Checkpoint::get_matrix(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end())
        throw InvalidArgumentError("Key not found in checkpoint: " + name);

    const auto& entry = it->second;
    if (entry.type != EntryType::matrix)
        throw InvalidArgumentError("Entry is not a matrix: " + name);
    if (entry.shape.size() != 2)
        throw InvalidArgumentError("Matrix entry must have 2 dimensions: " + name);

    Matrix mat(entry.shape[0], entry.shape[1]);
    checkCuda(cudaMemcpyAsync(mat.device_data(), entry.data.data(),
                              entry.data.size() * sizeof(float), cudaMemcpyHostToDevice,
                              compute_stream()),
              "cudaMemcpyAsync H2D");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");
    mat.mark_host_stale();
    return mat;
}

Tensor Checkpoint::get_tensor(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end())
        throw InvalidArgumentError("Key not found in checkpoint: " + name);

    const auto& entry = it->second;
    if (entry.type != EntryType::tensor)
        throw InvalidArgumentError("Entry is not a tensor: " + name);

    Tensor tensor(entry.shape);
    checkCuda(cudaMemcpyAsync(tensor.device_data(), entry.data.data(),
                              entry.data.size() * sizeof(float), cudaMemcpyHostToDevice,
                              compute_stream()),
              "cudaMemcpyAsync H2D");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");
    tensor.mark_host_stale();
    return tensor;
}

std::vector<std::string> Checkpoint::keys() const {
    std::vector<std::string> k;
    for (const auto& pair : entries_)
        k.push_back(pair.first);
    return k;
}

bool Checkpoint::contains(const std::string& name) const {
    return entries_.find(name) != entries_.end();
}

std::size_t Checkpoint::size() const {
    return entries_.size();
}

void Checkpoint::set_metadata(const std::string& key, const std::string& value) {
    metadata_[key] = value;
}

std::string Checkpoint::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end())
        throw InvalidArgumentError("Metadata key not found: " + key);
    return it->second;
}

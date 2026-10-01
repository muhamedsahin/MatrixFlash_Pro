// Dahili uygulama parçası: src/core/serialization.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Checkpoint girdilerini ve metadata bilgisini diske yazar.

// Checkpoint Implementation
void Checkpoint::add(const std::string& name, const Matrix& matrix) {
    Entry entry;
    entry.type = EntryType::matrix;
    entry.shape = {matrix.rows(), matrix.cols()};
    entry.data.resize(matrix.size());

    checkCuda(cudaMemcpyAsync(entry.data.data(), matrix.device_data(),
                              matrix.size() * sizeof(float), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");

    entries_[name] = std::move(entry);
}

void Checkpoint::add(const std::string& name, const Tensor& tensor) {
    Entry entry;
    entry.type = EntryType::tensor;
    entry.shape = tensor.shape();
    entry.data.resize(tensor.size());

    checkCuda(cudaMemcpyAsync(entry.data.data(), tensor.device_data(),
                              tensor.size() * sizeof(float), cudaMemcpyDeviceToHost,
                              compute_stream()),
              "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(compute_stream()), "cudaStreamSynchronize");

    entries_[name] = std::move(entry);
}

void Checkpoint::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw IoError("Failed to open file for writing: " + filename);

    uint32_t magic = MFC_MAGIC;
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint32_t num_metadata = static_cast<uint32_t>(metadata_.size());
    out.write(reinterpret_cast<const char*>(&num_metadata), sizeof(num_metadata));

    for (const auto& [k, v] : metadata_) {
        uint32_t k_len = static_cast<uint32_t>(k.size());
        out.write(reinterpret_cast<const char*>(&k_len), sizeof(k_len));
        out.write(k.data(), k_len);

        uint32_t v_len = static_cast<uint32_t>(v.size());
        out.write(reinterpret_cast<const char*>(&v_len), sizeof(v_len));
        out.write(v.data(), v_len);
    }

    uint32_t num_entries = static_cast<uint32_t>(entries_.size());
    out.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));

    for (const auto& [name, entry] : entries_) {
        uint32_t name_len = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(name.data(), name_len);

        uint8_t type_val = static_cast<uint8_t>(entry.type);
        out.write(reinterpret_cast<const char*>(&type_val), sizeof(type_val));

        uint32_t num_dims = static_cast<uint32_t>(entry.shape.size());
        out.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));

        std::vector<uint64_t> dims(num_dims);
        for (uint32_t i = 0; i < num_dims; ++i) {
            dims[i] = static_cast<uint64_t>(entry.shape[i]);
        }
        out.write(reinterpret_cast<const char*>(dims.data()), num_dims * sizeof(uint64_t));

        uint64_t data_size = static_cast<uint64_t>(entry.data.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        out.write(reinterpret_cast<const char*>(entry.data.data()), data_size);
    }

    if (!out.good()) {
        throw IoError("Error writing checkpoint to file: " + filename);
    }
}

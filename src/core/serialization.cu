#include "matrix_pro/core/serialization.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/utils/cuda_utils.hpp"
#include <fstream>
#include <cstring>
#include <cstdint>

namespace matrix_pro {

// File format magic numbers
constexpr uint32_t MFT_MAGIC = 0x5054464D; // 'M','F','T','P'
constexpr uint32_t MFC_MAGIC = 0x4B43464D; // 'M','F','C','K'

void save_tensor(const Tensor& tensor, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw IoError("Failed to open file for writing: " + filename);

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
    
    checkCuda(cudaMemcpyAsync(host_data.data(), tensor.data(), data_size, cudaMemcpyDeviceToHost, detail::compute_stream()), "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(detail::compute_stream()), "cudaStreamSynchronize");

    out.write(reinterpret_cast<const char*>(host_data.data()), data_size);
    if (!out.good()) {
        throw IoError("Failed to write tensor data to file: " + filename);
    }
}

Tensor load_tensor(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw IoError("Failed to open file for reading: " + filename);

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
    checkCuda(cudaMemcpyAsync(tensor.data(), host_data.data(), data_size, cudaMemcpyHostToDevice, detail::compute_stream()), "cudaMemcpyAsync H2D");
    checkCuda(cudaStreamSynchronize(detail::compute_stream()), "cudaStreamSynchronize");

    return tensor;
}

// Checkpoint Implementation
void Checkpoint::add(const std::string& name, const Matrix& matrix) {
    Entry entry;
    entry.type = EntryType::matrix;
    entry.shape = { matrix.rows(), matrix.cols() };
    entry.data.resize(matrix.size());
    
    checkCuda(cudaMemcpyAsync(entry.data.data(), matrix.data(), matrix.size() * sizeof(float), cudaMemcpyDeviceToHost, detail::compute_stream()), "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(detail::compute_stream()), "cudaStreamSynchronize");
    
    entries_[name] = std::move(entry);
}

void Checkpoint::add(const std::string& name, const Tensor& tensor) {
    Entry entry;
    entry.type = EntryType::tensor;
    entry.shape = tensor.shape();
    entry.data.resize(tensor.size());
    
    checkCuda(cudaMemcpyAsync(entry.data.data(), tensor.data(), tensor.size() * sizeof(float), cudaMemcpyDeviceToHost, detail::compute_stream()), "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(detail::compute_stream()), "cudaStreamSynchronize");
    
    entries_[name] = std::move(entry);
}

void Checkpoint::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw IoError("Failed to open file for writing: " + filename);

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

Checkpoint Checkpoint::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw IoError("Failed to open file for reading: " + filename);

    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != MFC_MAGIC || version != 1) {
        throw InvalidArgumentError("Invalid checkpoint format or version: " + filename);
    }

    Checkpoint cp;

    uint32_t num_metadata;
    in.read(reinterpret_cast<char*>(&num_metadata), sizeof(num_metadata));

    for (uint32_t i = 0; i < num_metadata; ++i) {
        uint32_t k_len;
        in.read(reinterpret_cast<char*>(&k_len), sizeof(k_len));
        std::string k(k_len, '\0');
        in.read(&k[0], k_len);
        
        uint32_t v_len;
        in.read(reinterpret_cast<char*>(&v_len), sizeof(v_len));
        std::string v(v_len, '\0');
        in.read(&v[0], v_len);
        
        cp.set_metadata(k, v);
    }

    uint32_t num_entries;
    in.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));

    for (uint32_t i = 0; i < num_entries; ++i) {
        uint32_t name_len;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        uint8_t type_val;
        in.read(reinterpret_cast<char*>(&type_val), sizeof(type_val));
        
        uint32_t num_dims;
        in.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));

        std::vector<uint64_t> dims(num_dims);
        in.read(reinterpret_cast<char*>(dims.data()), num_dims * sizeof(uint64_t));

        Entry entry;
        entry.type = static_cast<EntryType>(type_val);
        entry.shape.resize(num_dims);
        for (uint32_t j = 0; j < num_dims; ++j) {
            entry.shape[j] = static_cast<size_t>(dims[j]);
        }

        uint64_t data_size;
        in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        
        size_t num_elements = static_cast<size_t>(data_size / sizeof(float));
        entry.data.resize(num_elements);
        in.read(reinterpret_cast<char*>(entry.data.data()), data_size);

        cp.entries_[name] = std::move(entry);
    }

    if (!in.good()) {
        throw IoError("Error reading checkpoint from file: " + filename);
    }

    return cp;
}

Matrix Checkpoint::get_matrix(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) throw InvalidArgumentError("Key not found in checkpoint: " + name);
    
    const auto& entry = it->second;
    if (entry.type != EntryType::matrix) throw InvalidArgumentError("Entry is not a matrix: " + name);
    if (entry.shape.size() != 2) throw InvalidArgumentError("Matrix entry must have 2 dimensions: " + name);
    
    Matrix mat(entry.shape[0], entry.shape[1]);
    checkCuda(cudaMemcpyAsync(mat.data(), entry.data.data(), entry.data.size() * sizeof(float), cudaMemcpyHostToDevice, detail::compute_stream()), "cudaMemcpyAsync H2D");
    checkCuda(cudaStreamSynchronize(detail::compute_stream()), "cudaStreamSynchronize");
    return mat;
}

Tensor Checkpoint::get_tensor(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) throw InvalidArgumentError("Key not found in checkpoint: " + name);
    
    const auto& entry = it->second;
    if (entry.type != EntryType::tensor) throw InvalidArgumentError("Entry is not a tensor: " + name);
    
    Tensor tensor(entry.shape);
    checkCuda(cudaMemcpyAsync(tensor.data(), entry.data.data(), entry.data.size() * sizeof(float), cudaMemcpyHostToDevice, detail::compute_stream()), "cudaMemcpyAsync H2D");
    checkCuda(cudaStreamSynchronize(detail::compute_stream()), "cudaStreamSynchronize");
    return tensor;
}

std::vector<std::string> Checkpoint::keys() const {
    std::vector<std::string> k;
    for (const auto& pair : entries_) k.push_back(pair.first);
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
    if (it == metadata_.end()) throw InvalidArgumentError("Metadata key not found: " + key);
    return it->second;
}

} // namespace matrix_pro

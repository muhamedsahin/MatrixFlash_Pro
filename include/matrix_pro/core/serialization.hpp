#pragma once
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace matrix_pro {

class Matrix;
class Tensor;
class SparseCSR;

// Single tensor save/load
void save_tensor(const Tensor& tensor, const std::string& filename);
Tensor load_tensor(const std::string& filename);

// Multi-tensor checkpoint
class Checkpoint {
public:
    Checkpoint() = default;

    void add(const std::string& name, const Matrix& matrix);
    void add(const std::string& name, const Tensor& tensor);

    void save(const std::string& filename) const;
    static Checkpoint load(const std::string& filename);

    Matrix get_matrix(const std::string& name) const;
    Tensor get_tensor(const std::string& name) const;

    std::vector<std::string> keys() const;
    bool contains(const std::string& name) const;
    std::size_t size() const;

    void set_metadata(const std::string& key, const std::string& value);
    std::string get_metadata(const std::string& key) const;

private:
    enum class EntryType : std::uint8_t { matrix = 0, tensor = 1 };
    struct Entry {
        EntryType type;
        std::vector<std::size_t> shape;  // rows,cols for matrix; shape for tensor
        std::vector<float> data;
    };
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::string> metadata_;
};

}

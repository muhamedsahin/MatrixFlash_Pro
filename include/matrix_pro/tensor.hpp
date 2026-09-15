#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace matrix_pro {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);
    Tensor(std::vector<std::size_t> shape, const std::vector<float>& values);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    std::size_t rank() const noexcept { return shape_.size(); }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    const std::vector<float>& data() const noexcept { return host_data_; }
    std::vector<float>& data() noexcept { return host_data_; }
    float* device_data() noexcept { return device_data_.get(); }
    const float* device_data() const noexcept { return device_data_.get(); }

    void upload();
    void download();
    void fill(float value);
    void synchronize() const;

private:
    std::vector<std::size_t> shape_;
    std::size_t size_ = 0;
    std::vector<float> host_data_;
    std::unique_ptr<float, void (*)(float*)> device_data_{nullptr, nullptr};

    void allocate_device();
};

}
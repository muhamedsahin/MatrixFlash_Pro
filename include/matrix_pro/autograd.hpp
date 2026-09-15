#pragma once

#include <memory>

#include "matrix.hpp"

namespace matrix_pro {

class Variable {
public:
    struct Node;

    Variable() = default;
    explicit Variable(const Matrix& value, bool requires_grad = true);

    const Matrix& value() const;
    const Matrix& grad() const;
    bool requires_grad() const;
    void zero_grad();
    void backward();

    Variable add(const Variable& other) const;
    Variable matmul(const Variable& other) const;
    Variable relu() const;
    Variable leaky_relu(float negative_slope = 0.01f) const;
    Variable gelu() const;
    Variable swish(float beta = 1.0f) const;

private:
    std::shared_ptr<Node> node_;
    explicit Variable(std::shared_ptr<Node> node);
};

}
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"

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
    Variable subtract(const Variable& other) const;
    Variable elementwise_multiply(const Variable& other) const;
    Variable divide(const Variable& other) const;
    Variable matmul(const Variable& other) const;
    Variable multiply(float scalar) const;
    Variable add_scalar(float scalar) const;
    Variable negate() const;
    Variable pow(float exponent) const;

    Variable add_row_vector(const Variable& vector) const;
    Variable add_col_vector(const Variable& vector) const;
    Variable multiply_row_vector(const Variable& vector) const;
    Variable multiply_col_vector(const Variable& vector) const;
    // General 2D broadcasting ((M,1) with (1,N) etc); backward sums the gradient
    // back over whichever dimensions were broadcast.
    Variable broadcast_add(const Variable& other) const;
    Variable broadcast_multiply(const Variable& other) const;

    // Concatenates variables along axis 0 (rows) or 1 (cols). The backward pass
    // slices the upstream gradient back into each operand.
    static Variable concat(const std::vector<Variable>& parts, std::size_t axis);

    Variable transpose() const;
    Variable flatten() const;
    Variable reshape(std::size_t rows, std::size_t cols) const;
    Variable slice(std::size_t row_start, std::size_t row_end,
                   std::size_t col_start, std::size_t col_end) const;

    Variable relu() const;
    Variable leaky_relu(float negative_slope = 0.01f) const;
    Variable gelu() const;
    Variable swish(float beta = 1.0f) const;
    Variable sigmoid() const;
    Variable tanh() const;
    Variable elu(float alpha = 1.0f) const;
    Variable softplus() const;
    Variable mish() const;
    Variable hardtanh(float low = -1.0f, float high = 1.0f) const;
    Variable hardsigmoid() const;
    Variable hardswish() const;
    Variable selu() const;
    // Learnable-parameter activation: `alpha` must itself require gradients.
    Variable prelu(const Variable& alpha) const;

    // Generic custom elementwise op. forward maps input -> output on the GPU
    // (any composition of Matrix primitives, including user CUDA kernels);
    // backward receives (grad_output, input, output) and must return dL/dinput.
    Variable custom_unary(const std::string& name,
                          std::function<Matrix(const Matrix&)> forward,
                          std::function<Matrix(const Matrix&, const Matrix&, const Matrix&)> backward) const;

private:
    std::shared_ptr<Node> node_;
    explicit Variable(std::shared_ptr<Node> node);

    friend Variable custom_loss(const Variable&, const Variable&,
                                std::function<float(const Matrix&, const Matrix&)>,
                                std::function<Matrix(const Matrix&, const Matrix&)>);
};

// Generic custom loss. forward(pred, target) returns a scalar loss value;
// backward(pred, target) returns dL/dprediction (upstream scaling is applied
// internally). Both run entirely from Matrix primitives / user kernels.
Variable custom_loss(const Variable& prediction, const Variable& target,
                     std::function<float(const Matrix&, const Matrix&)> forward,
                     std::function<Matrix(const Matrix&, const Matrix&)> backward);

Variable mse_loss(const Variable& prediction, const Variable& target);
Variable mae_loss(const Variable& prediction, const Variable& target);   // subgradient 0 at |d| = 0
Variable huber_loss(const Variable& prediction, const Variable& target, float delta = 1.0f);
Variable bce_with_logits_loss(const Variable& logits, const Variable& target);       // fused sigmoid+BCE
Variable softmax_cross_entropy_loss(const Variable& logits, const Variable& one_hot_target); // fused
Variable kl_divergence_loss(const Variable& prediction, const Variable& target);

// Autograd-enabled tensor. Shares the tape machinery with Variable but stores
// rank-N data; CNN ops (conv2d, pooling) and batched matmul are differentiable.
class VarTensor {
public:
    struct Node;

    VarTensor() = default;
    explicit VarTensor(const Tensor& value, bool requires_grad = true);

    const Tensor& value() const;
    const Tensor& grad() const;
    bool requires_grad() const;
    void zero_grad();
    void backward();

    VarTensor add(const VarTensor& other) const;
    VarTensor elementwise_multiply(const VarTensor& other) const;
    VarTensor multiply(float scalar) const;
    VarTensor negate() const;
    VarTensor batch_matmul(const VarTensor& other) const;

    // CNN ops; weights [K,C,R,S], bias [K] (all differentiable).
    VarTensor conv2d(const VarTensor& weights, const VarTensor& bias,
                     std::size_t stride = 1, std::size_t padding = 0) const;
    VarTensor max_pool2d(std::size_t kernel_size, std::size_t stride = 1, std::size_t padding = 0) const;
    VarTensor avg_pool2d(std::size_t kernel_size, std::size_t stride = 1, std::size_t padding = 0) const;

    VarTensor custom_unary(const std::string& name,
                           std::function<Tensor(const Tensor&)> forward,
                           std::function<Tensor(const Tensor&, const Tensor&, const Tensor&)> backward) const;

private:
    std::shared_ptr<Node> node_;
    explicit VarTensor(std::shared_ptr<Node> node);

    friend VarTensor custom_loss(const VarTensor&, const VarTensor&,
                                 std::function<float(const Tensor&, const Tensor&)>,
                                 std::function<Tensor(const Tensor&, const Tensor&)>);
};

VarTensor custom_loss(const VarTensor& prediction, const VarTensor& target,
                      std::function<float(const Tensor&, const Tensor&)> forward,
                      std::function<Tensor(const Tensor&, const Tensor&)> backward);

}

#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/ops/operations.hpp"

#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace matrix_pro {

struct Variable::Node {
    Matrix value;
    Matrix gradient;
    bool requires_gradient = false;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward_function;
};

struct VarTensor::Node {
    Tensor value;
    Tensor gradient;
    bool requires_gradient = false;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward_function;
};

namespace {

std::shared_ptr<Variable::Node> make_node(const Matrix& value, bool requires_gradient) {
    auto node = std::make_shared<Variable::Node>();
    node->value = value;
    node->requires_gradient = requires_gradient;
    if (requires_gradient) node->gradient = Matrix::zeros(value.rows(), value.cols());
    return node;
}

std::shared_ptr<VarTensor::Node> make_node(const Tensor& value, bool requires_gradient) {
    auto node = std::make_shared<VarTensor::Node>();
    node->value = value;
    node->requires_gradient = requires_gradient;
    if (requires_gradient) {
        node->gradient = Tensor(value.shape());
        node->gradient.zero();   // accumulators, never garbage
    }
    return node;
}

void collect_nodes(const std::shared_ptr<Variable::Node>& node,
                   std::unordered_set<Variable::Node*>& visited,
                   std::vector<std::shared_ptr<Variable::Node>>& order) {
    if (!node || !visited.insert(node.get()).second) return;
    for (const auto& parent : node->parents) collect_nodes(parent, visited, order);
    order.push_back(node);
}

void collect_nodes(const std::shared_ptr<VarTensor::Node>& node,
                   std::unordered_set<VarTensor::Node*>& visited,
                   std::vector<std::shared_ptr<VarTensor::Node>>& order) {
    if (!node || !visited.insert(node.get()).second) return;
    for (const auto& parent : node->parents) collect_nodes(parent, visited, order);
    order.push_back(node);
}

Matrix scalar_matrix(float value) { return Matrix(1, 1, std::vector<float>{value}); }

Matrix band_mask(const Matrix& m, float low, float high) {
    return matrix_pro::elementwise_multiply(m.greater(low), m.less(high));
}

}

Variable::Variable(const Matrix& value, bool requires_grad)
    : node_(make_node(value, requires_grad)) {}

Variable::Variable(std::shared_ptr<Node> node) : node_(std::move(node)) {}

const Matrix& Variable::value() const {
    if (!node_) throw MatrixProError("Empty autograd variable");
    return node_->value;
}

const Matrix& Variable::grad() const {
    if (!node_) throw MatrixProError("Empty autograd variable");
    return node_->gradient;
}

bool Variable::requires_grad() const { return node_ && node_->requires_gradient; }

void Variable::zero_grad() {
    if (requires_grad()) node_->gradient = Matrix::zeros(value().rows(), value().cols());
}

void Variable::backward() {
    if (!node_) throw MatrixProError("Cannot backward an empty variable");
    if (!node_->requires_gradient) throw InvalidArgumentError("Root variable does not require gradients");
    node_->gradient = Matrix::ones(value().rows(), value().cols());
    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;
    collect_nodes(node_, visited, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_function) (*it)->backward_function();
    }
}

// --- arithmetic ---

Variable Variable::add(const Variable& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot add empty variables");
    auto result = make_node(matrix_pro::add(value(), other.value()), requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient) left->gradient += self->gradient;
        if (right->requires_gradient) right->gradient += self->gradient;
    };
    return Variable(result);
}

Variable Variable::subtract(const Variable& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot subtract empty variables");
    auto result = make_node(matrix_pro::subtract(value(), other.value()), requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient) left->gradient += self->gradient;
        if (right->requires_gradient) right->gradient -= self->gradient;
    };
    return Variable(result);
}

Variable Variable::elementwise_multiply(const Variable& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot multiply empty variables");
    auto result = make_node(matrix_pro::elementwise_multiply(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient += matrix_pro::elementwise_multiply(self->gradient, right->value);
        if (right->requires_gradient)
            right->gradient += matrix_pro::elementwise_multiply(self->gradient, left->value);
    };
    return Variable(result);
}

Variable Variable::divide(const Variable& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot divide empty variables");
    auto result = make_node(matrix_pro::divide(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        // d/dA = g / B ;  d/dB = -g * A / B^2
        if (left->requires_gradient)
            left->gradient += matrix_pro::divide(self->gradient, right->value);
        if (right->requires_gradient) {
            Matrix quotient = matrix_pro::divide(matrix_pro::elementwise_multiply(self->gradient, left->value),
                                                 matrix_pro::elementwise_multiply(right->value, right->value));
            right->gradient -= quotient;
        }
    };
    return Variable(result);
}

Variable Variable::matmul(const Variable& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot multiply empty variables");
    auto result = make_node(matrix_pro::multiply(value(), other.value()), requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient) left->gradient += matrix_pro::multiply(self->gradient, matrix_pro::transpose(right->value));
        if (right->requires_gradient) right->gradient += matrix_pro::multiply(matrix_pro::transpose(left->value), self->gradient);
    };
    return Variable(result);
}

Variable Variable::multiply(float scalar) const {
    if (!node_) throw MatrixProError("Cannot scale an empty variable");
    auto result = make_node(value() * scalar, requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, scalar]() {
        if (input->requires_gradient) input->gradient += self->gradient * scalar;
    };
    return Variable(result);
}

Variable Variable::add_scalar(float scalar) const {
    if (!node_) throw MatrixProError("Cannot shift an empty variable");
    auto result = make_node(value().add_scalar(scalar), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) input->gradient += self->gradient;
    };
    return Variable(result);
}

Variable Variable::negate() const { return multiply(-1.0f); }

Variable Variable::pow(float exponent) const {
    if (!node_) throw MatrixProError("Cannot exponentiate an empty variable");
    auto result = make_node(value().pow(exponent), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, exponent]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, input->value.pow(exponent - 1.0f) * exponent);
    };
    return Variable(result);
}

// --- broadcast (bias-style) ops ---

Variable Variable::add_row_vector(const Variable& vector) const {
    if (!node_ || !vector.node_) throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::add_row_vector(value(), vector.value()),
                            requires_grad() || vector.requires_grad());
    result->parents = {node_, vector.node_};
    result->backward_function = [self = result.get(), input = node_, vector = vector.node_]() {
        if (input->requires_gradient) input->gradient += self->gradient;
        if (vector->requires_gradient) vector->gradient += matrix_pro::broadcast_backward(self->gradient, vector->value);
    };
    return Variable(result);
}

Variable Variable::add_col_vector(const Variable& vector) const {
    if (!node_ || !vector.node_) throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::add_col_vector(value(), vector.value()),
                            requires_grad() || vector.requires_grad());
    result->parents = {node_, vector.node_};
    result->backward_function = [self = result.get(), input = node_, vector = vector.node_]() {
        if (input->requires_gradient) input->gradient += self->gradient;
        if (vector->requires_gradient) vector->gradient += matrix_pro::broadcast_backward(self->gradient, vector->value);
    };
    return Variable(result);
}

Variable Variable::multiply_row_vector(const Variable& vector) const {
    if (!node_ || !vector.node_) throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::multiply_row_vector(value(), vector.value()),
                            requires_grad() || vector.requires_grad());
    result->parents = {node_, vector.node_};
    result->backward_function = [self = result.get(), input = node_, vector = vector.node_]() {
        // out = M * v[row-broadcast]; dM = g * v ; dv = sum over rows of g * M
        if (input->requires_gradient)
            input->gradient += matrix_pro::broadcast_multiply(self->gradient, vector->value);
        if (vector->requires_gradient)
            vector->gradient += matrix_pro::broadcast_backward(
                matrix_pro::elementwise_multiply(self->gradient, input->value), vector->value);
    };
    return Variable(result);
}

Variable Variable::multiply_col_vector(const Variable& vector) const {
    if (!node_ || !vector.node_) throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::multiply_col_vector(value(), vector.value()),
                            requires_grad() || vector.requires_grad());
    result->parents = {node_, vector.node_};
    result->backward_function = [self = result.get(), input = node_, vector = vector.node_]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::broadcast_multiply(self->gradient, vector->value);
        if (vector->requires_gradient)
            vector->gradient += matrix_pro::broadcast_backward(
                matrix_pro::elementwise_multiply(self->gradient, input->value), vector->value);
    };
    return Variable(result);
}

// --- general broadcasting & concatenation ---

Variable Variable::broadcast_add(const Variable& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::broadcast_add(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient += matrix_pro::broadcast_backward(self->gradient, left->value);
        if (right->requires_gradient)
            right->gradient += matrix_pro::broadcast_backward(self->gradient, right->value);
    };
    return Variable(result);
}

Variable Variable::broadcast_multiply(const Variable& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::broadcast_multiply(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        // d(out)/d(left) = right, reduced back to left's shape; d(out)/d(right) = left.
        if (left->requires_gradient)
            left->gradient += matrix_pro::broadcast_backward(
                matrix_pro::broadcast_multiply(self->gradient, right->value), left->value);
        if (right->requires_gradient)
            right->gradient += matrix_pro::broadcast_backward(
                matrix_pro::broadcast_multiply(self->gradient, left->value), right->value);
    };
    return Variable(result);
}

Variable Variable::concat(const std::vector<Variable>& parts, std::size_t axis) {
    if (parts.empty()) throw InvalidArgumentError("concat requires at least one variable");
    if (parts.size() == 1) return parts.front();
    std::vector<Matrix> values;
    std::vector<std::shared_ptr<Node>> parents;
    values.reserve(parts.size());
    parents.reserve(parts.size());
    bool requires = false;
    for (const Variable& part : parts) {
        if (!part.node_) throw MatrixProError("Cannot concat empty variables");
        values.push_back(part.value());
        parents.push_back(part.node_);
        requires = requires || part.requires_grad();
    }
    auto result = make_node(matrix_pro::concat(values, axis), requires);
    result->parents = parents;
    result->backward_function = [self = result.get(), parents, axis]() {
        std::size_t offset = 0;
        for (const auto& parent : parents) {
            const std::size_t rows = parent->value.rows();
            const std::size_t cols = parent->value.cols();
            if (parent->requires_gradient) {
                // Each operand simply receives the slice of the upstream gradient
                // that it contributed during the forward pass.
                parent->gradient += axis == 0
                    ? matrix_pro::slice(self->gradient, offset, offset + rows, 0, cols)
                    : matrix_pro::slice(self->gradient, 0, rows, offset, offset + cols);
            }
            offset += axis == 0 ? rows : cols;
        }
    };
    return Variable(result);
}

// --- transforms ---

Variable Variable::transpose() const {
    if (!node_) throw MatrixProError("Cannot transpose an empty variable");
    auto result = make_node(value().transpose(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) input->gradient += self->gradient.transpose();
    };
    return Variable(result);
}

Variable Variable::flatten() const {
    if (!node_) throw MatrixProError("Cannot flatten an empty variable");
    const std::size_t rows = value().rows(), cols = value().cols();
    auto result = make_node(value().flatten(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, rows, cols]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::reshape(self->gradient, rows, cols);
    };
    return Variable(result);
}

Variable Variable::reshape(std::size_t rows, std::size_t cols) const {
    if (!node_) throw ShapeMismatchError("Cannot reshape an empty variable");
    const std::size_t original_rows = value().rows(), original_cols = value().cols();
    auto result = make_node(matrix_pro::reshape(value(), rows, cols), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, original_rows, original_cols]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::reshape(self->gradient, original_rows, original_cols);
    };
    return Variable(result);
}

Variable Variable::slice(std::size_t row_start, std::size_t row_end,
                         std::size_t col_start, std::size_t col_end) const {
    if (!node_) throw OutOfRangeError("Cannot slice an empty variable");
    auto result = make_node(value().slice(row_start, row_end, col_start, col_end), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, row_start, row_end, col_start, col_end]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::slice_scatter(self->gradient, input->value,
                                                         row_start, row_end, col_start, col_end);
    };
    return Variable(result);
}

// --- activations ---

Variable Variable::relu() const {
    auto result = make_node(value().relu(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) input->gradient += matrix_pro::elementwise_multiply(self->gradient, input->value.greater(0.0f));
    };
    return Variable(result);
}

Variable Variable::leaky_relu(float negative_slope) const {
    auto result = make_node(value().leaky_relu(negative_slope), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, negative_slope]() {
        if (input->requires_gradient) {
            Matrix derivative = matrix_pro::where(input->value.greater(0.0f), 1.0f, negative_slope);
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::sigmoid() const {
    auto result = make_node(value().sigmoid(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) {
            // s' = s * (1 - s), read straight from the saved output.
            const Matrix& s = self->value;
            Matrix derivative = matrix_pro::elementwise_multiply(s, Matrix::ones(s.rows(), s.cols()) - s);
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::tanh() const {
    auto result = make_node(value().tanh(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) {
            const Matrix& t = self->value;
            Matrix derivative = Matrix::ones(t.rows(), t.cols()) - matrix_pro::elementwise_multiply(t, t);
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::gelu() const {
    auto result = make_node(value().gelu(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) {
            const float inverse_sqrt_two_pi = 0.7978845608f;
            Matrix cdf = (input->value + input->value.pow(3.0f) * 0.044715f) * inverse_sqrt_two_pi;
            Matrix tanh_value = cdf.tanh();
            Matrix term = tanh_value.add_scalar(1.0f);
            Matrix derivative = term * 0.5f;
            Matrix du_dx = (Matrix::ones(input->value.rows(), input->value.cols()) + input->value.pow(2.0f) * (3.0f * 0.044715f)) * inverse_sqrt_two_pi;
            derivative += matrix_pro::elementwise_multiply(input->value, matrix_pro::elementwise_multiply(Matrix::ones(input->value.rows(), input->value.cols()) - matrix_pro::elementwise_multiply(tanh_value, tanh_value), du_dx)) * 0.5f;
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::swish(float beta) const {
    auto result = make_node(value().swish(beta), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, beta]() {
        if (input->requires_gradient) {
            Matrix sigmoid = (input->value * beta).sigmoid();
            Matrix derivative = sigmoid + matrix_pro::elementwise_multiply(input->value * beta,
                matrix_pro::elementwise_multiply(sigmoid, Matrix::ones(input->value.rows(), input->value.cols()) - sigmoid));
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::elu(float alpha) const {
    auto result = make_node(value().elu(alpha), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, alpha]() {
        if (input->requires_gradient) {
            Matrix positive = Matrix::ones(input->value.rows(), input->value.cols());
            Matrix negative = input->value.exp() * alpha;
            Matrix derivative = matrix_pro::where(input->value.greater(0.0f), positive, negative);
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::softplus() const {
    auto result = make_node(matrix_pro::softplus(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::softplus_backward(input->value, self->gradient);
    };
    return Variable(result);
}

Variable Variable::mish() const {
    auto result = make_node(matrix_pro::mish(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::mish_backward(input->value, self->gradient);
    };
    return Variable(result);
}

Variable Variable::hardtanh(float low, float high) const {
    auto result = make_node(matrix_pro::hardtanh(value(), low, high), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, low, high]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, band_mask(input->value, low, high));
    };
    return Variable(result);
}

Variable Variable::hardsigmoid() const {
    auto result = make_node(matrix_pro::hardsigmoid(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, band_mask(input->value, -3.0f, 3.0f) * (1.0f / 6.0f));
    };
    return Variable(result);
}

Variable Variable::hardswish() const {
    auto result = make_node(matrix_pro::hardswish(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) {
            // d/dx = (x/3 + 0.5) inside (-3, 3), 1 for x > 3, 0 for x < -3.
            const Matrix& x = input->value;
            Matrix inner = matrix_pro::elementwise_multiply(band_mask(x, -3.0f, 3.0f), (x * (1.0f / 3.0f)).add_scalar(0.5f));
            Matrix derivative = matrix_pro::where(x.greater(3.0f), Matrix::ones(x.rows(), x.cols()), inner);
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::selu() const {
    auto result = make_node(matrix_pro::selu(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient) {
            const float lambda = 1.0507009873554805f;
            const float alpha = 1.6732632423543772f;
            Matrix positive = Matrix::ones(input->value.rows(), input->value.cols()) * lambda;
            Matrix negative = input->value.exp() * (lambda * alpha);
            Matrix derivative = matrix_pro::where(input->value.greater(0.0f), positive, negative);
            input->gradient += matrix_pro::elementwise_multiply(self->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::prelu(const Variable& alpha) const {
    if (!node_ || !alpha.node_) throw MatrixProError("Cannot apply PReLU to empty variables");
    auto result = make_node(matrix_pro::prelu(value(), alpha.value()),
                            requires_grad() || alpha.requires_grad());
    result->parents = {node_, alpha.node_};
    result->backward_function = [self = result.get(), input = node_, alpha = alpha.node_]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::where(input->value.greater(0.0f), self->gradient,
                                                 matrix_pro::broadcast_multiply(self->gradient, alpha->value));
        if (alpha->requires_gradient) {
            // dL/dalpha = sum over elements of x * g on the non-positive side.
            // For scalar (1x1) alpha, accumulate the sum. For per-element alpha,
            // the contribution must match alpha's shape.
            Matrix negative_side = matrix_pro::logical_not(input->value.greater(0.0f));
            Matrix contribution = matrix_pro::elementwise_multiply(
                input->value, matrix_pro::elementwise_multiply(self->gradient, negative_side));
            if (alpha->value.rows() == 1 && alpha->value.cols() == 1) {
                alpha->gradient += scalar_matrix(contribution.sum());
            } else {
                // Per-element alpha: alpha->value has same shape as input (enforced by forward),
                // so contribution (same shape as input) can be accumulated directly.
                if (contribution.rows() == alpha->value.rows() && contribution.cols() == alpha->value.cols())
                    alpha->gradient += contribution;
                else
                    throw ShapeMismatchError("PReLU backward: alpha gradient shape mismatch");
            }
        }
    };
    return Variable(result);
}

Variable Variable::custom_unary(const std::string& name,
                                std::function<Matrix(const Matrix&)> forward,
                                std::function<Matrix(const Matrix&, const Matrix&, const Matrix&)> backward) const {
    if (!node_) throw MatrixProError("Cannot apply a custom op to an empty variable");
    if (!forward || !backward) throw InvalidArgumentError("custom_unary requires forward and backward for " + name);
    auto result = make_node(forward(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, backward]() {
        if (input->requires_gradient)
            input->gradient += backward(self->gradient, input->value, self->value);
    };
    return Variable(result);
}

// --- custom & built-in losses ---

Variable custom_loss(const Variable& prediction, const Variable& target,
                     std::function<float(const Matrix&, const Matrix&)> forward,
                     std::function<Matrix(const Matrix&, const Matrix&)> backward) {
    if (!prediction.node_ || !target.node_) throw MatrixProError("custom_loss requires non-empty variables");
    if (!forward || !backward) throw InvalidArgumentError("custom_loss requires forward and backward");
    auto result = make_node(scalar_matrix(forward(prediction.value(), target.value())),
                            prediction.requires_grad() || target.requires_grad());
    result->parents = {prediction.node_, target.node_};
    result->backward_function = [self = result.get(), pred = prediction.node_, tgt = target.node_, backward]() {
        // dL/dprediction = upstream scalar (1x1) * backward(pred, tgt), applied
        // through broadcasting so the upstream gradient never leaves the GPU.
        if (pred->requires_gradient) {
            Matrix raw = backward(pred->value, tgt->value);
            pred->gradient += matrix_pro::broadcast_multiply(raw, self->gradient);
        }
        if (tgt->requires_gradient) {
            Matrix negated = matrix_pro::negate(backward(pred->value, tgt->value));
            tgt->gradient += matrix_pro::broadcast_multiply(negated, self->gradient);
        }
    };
    return Variable(result);
}

Variable mse_loss(const Variable& prediction, const Variable& target) {
    return custom_loss(prediction, target,
        [](const Matrix& p, const Matrix& t) {
            Matrix diff = matrix_pro::subtract(p, t);
            return sum(matrix_pro::elementwise_multiply(diff, diff)) / static_cast<float>(p.size());
        },
        [](const Matrix& p, const Matrix& t) {
            Matrix diff = matrix_pro::subtract(p, t);
            return (matrix_pro::elementwise_multiply(diff, Matrix::ones(p.rows(), p.cols()) * 2.0f)) * (1.0f / static_cast<float>(p.size()));
        });
}

Variable mae_loss(const Variable& prediction, const Variable& target) {
    return custom_loss(prediction, target,
        [](const Matrix& p, const Matrix& t) {
            return sum(matrix_pro::abs(matrix_pro::subtract(p, t))) / static_cast<float>(p.size());
        },
        [](const Matrix& p, const Matrix& t) {
            // sign(p - t) with subgradient 0 where p == t.
            Matrix diff = matrix_pro::subtract(p, t);
            Matrix ones = Matrix::ones(p.rows(), p.cols());
            Matrix sign = matrix_pro::where(diff.greater(0.0f), ones,
                                            matrix_pro::where(diff.less(0.0f), matrix_pro::negate(ones), Matrix::zeros(p.rows(), p.cols())));
            return sign * (1.0f / static_cast<float>(p.size()));
        });
}

Variable huber_loss(const Variable& prediction, const Variable& target, float delta) {
    return custom_loss(prediction, target,
        [delta](const Matrix& p, const Matrix& t) {
            Matrix diff = matrix_pro::subtract(p, t);
            Matrix abs_diff = matrix_pro::abs(diff);
            Matrix quadratic = matrix_pro::elementwise_multiply(abs_diff, abs_diff) * 0.5f;
            Matrix linear = abs_diff.add_scalar(-0.5f * delta) * delta;
            Matrix loss = matrix_pro::where(abs_diff.greater(delta), linear, quadratic);
            return sum(loss) / static_cast<float>(p.size());
        },
        [delta](const Matrix& p, const Matrix& t) {
            Matrix diff = matrix_pro::subtract(p, t);
            Matrix abs_diff = matrix_pro::abs(diff);
            Matrix ones = Matrix::ones(p.rows(), p.cols());
            Matrix sign = matrix_pro::where(diff.greater(0.0f), ones,
                                            matrix_pro::where(diff.less(0.0f), matrix_pro::negate(ones), Matrix::zeros(p.rows(), p.cols())));
            Matrix grad = matrix_pro::where(abs_diff.greater(delta), sign * delta, diff);
            return grad * (1.0f / static_cast<float>(p.size()));
        });
}

Variable bce_with_logits_loss(const Variable& logits, const Variable& target) {
    return custom_loss(logits, target,
        [](const Matrix& z, const Matrix& t) { return bce_with_logits_value(z, t); },
        [](const Matrix& z, const Matrix& t) { return bce_with_logits_grad(z, t); });
}

Variable softmax_cross_entropy_loss(const Variable& logits, const Variable& one_hot_target) {
    return custom_loss(logits, one_hot_target,
        [](const Matrix& z, const Matrix& t) {
            return sum(softmax_cross_entropy_value(z, t)) / static_cast<float>(z.rows());
        },
        [](const Matrix& z, const Matrix& t) { return softmax_cross_entropy_grad(z, t); });
}

Variable kl_divergence_loss(const Variable& prediction, const Variable& target) {
    return custom_loss(prediction, target,
        [](const Matrix& p, const Matrix& t) {
            // mean(t * (log t - log p)); t must be strictly positive.
            Matrix log_t = t.log();
            Matrix log_p = p.log();
            Matrix diff = matrix_pro::subtract(log_t, log_p);
            return sum(matrix_pro::elementwise_multiply(t, diff)) / static_cast<float>(p.size());
        },
        [](const Matrix& p, const Matrix& t) {
            Matrix grad = matrix_pro::divide(t, p) * (-1.0f);
            return grad * (1.0f / static_cast<float>(p.size()));
        });
}

// === VarTensor: autograd for rank-N tensors ===

VarTensor::VarTensor(const Tensor& value, bool requires_grad)
    : node_(make_node(value, requires_grad)) {}

VarTensor::VarTensor(std::shared_ptr<Node> node) : node_(std::move(node)) {}

const Tensor& VarTensor::value() const {
    if (!node_) throw MatrixProError("Empty autograd tensor");
    return node_->value;
}

const Tensor& VarTensor::grad() const {
    if (!node_) throw MatrixProError("Empty autograd tensor");
    return node_->gradient;
}

bool VarTensor::requires_grad() const { return node_ && node_->requires_gradient; }

void VarTensor::zero_grad() {
    if (requires_grad()) node_->gradient.zero();
}

void VarTensor::backward() {
    if (!node_) throw MatrixProError("Cannot backward an empty tensor variable");
    if (!node_->requires_gradient) throw InvalidArgumentError("Root tensor variable does not require gradients");
    node_->gradient = Tensor(value().shape());
    node_->gradient.fill(1.0f);
    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;
    collect_nodes(node_, visited, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_function) (*it)->backward_function();
    }
}

VarTensor VarTensor::add(const VarTensor& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot add empty tensor variables");
    auto result = make_node(tensor_add(value(), other.value()), requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient) left->gradient = tensor_add(left->gradient, self->gradient);
        if (right->requires_gradient) right->gradient = tensor_add(right->gradient, self->gradient);
    };
    return VarTensor(result);
}

VarTensor VarTensor::elementwise_multiply(const VarTensor& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot multiply empty tensor variables");
    auto result = make_node(tensor_multiply(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient = tensor_add(left->gradient, tensor_multiply(self->gradient, right->value));
        if (right->requires_gradient)
            right->gradient = tensor_add(right->gradient, tensor_multiply(self->gradient, left->value));
    };
    return VarTensor(result);
}

VarTensor VarTensor::multiply(float scalar) const {
    if (!node_) throw MatrixProError("Cannot scale an empty tensor variable");
    auto result = make_node(tensor_multiply(value(), scalar), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, scalar]() {
        if (input->requires_gradient)
            input->gradient = tensor_add(input->gradient, tensor_multiply(self->gradient, scalar));
    };
    return VarTensor(result);
}

VarTensor VarTensor::negate() const { return multiply(-1.0f); }

VarTensor VarTensor::batch_matmul(const VarTensor& other) const {
    if (!node_ || !other.node_) throw MatrixProError("Cannot batch-multiply empty tensor variables");
    auto result = make_node(matrix_pro::batch_matmul(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient = tensor_add(left->gradient, batch_matmul_backward_left(self->gradient, right->value));
        if (right->requires_gradient)
            right->gradient = tensor_add(right->gradient, batch_matmul_backward_right(self->gradient, left->value));
    };
    return VarTensor(result);
}

VarTensor VarTensor::conv2d(const VarTensor& weights, const VarTensor& bias,
                            std::size_t stride, std::size_t padding) const {
    if (!node_ || !weights.node_ || !bias.node_)
        throw MatrixProError("conv2d requires non-empty tensor variables");
    auto result = make_node(matrix_pro::conv2d(value(), weights.value(), bias.value(), stride, padding),
                            requires_grad() || weights.requires_grad() || bias.requires_grad());
    result->parents = {node_, weights.node_, bias.node_};
    result->backward_function = [self = result.get(), input = node_, weight = weights.node_, bias = bias.node_,
                                 stride, padding]() {
        if (input->requires_gradient)
            input->gradient = tensor_add(input->gradient,
                conv2d_input_backward(self->gradient, input->value, weight->value, stride, padding));
        if (weight->requires_gradient)
            weight->gradient = tensor_add(weight->gradient,
                conv2d_weight_backward(self->gradient, input->value, weight->value, stride, padding));
        if (bias->requires_gradient)
            bias->gradient = tensor_add(bias->gradient, conv2d_bias_backward(self->gradient));
    };
    return VarTensor(result);
}

VarTensor VarTensor::max_pool2d(std::size_t kernel_size, std::size_t stride, std::size_t padding) const {
    if (!node_) throw MatrixProError("Cannot pool an empty tensor variable");
    auto result = make_node(matrix_pro::max_pool2d(value(), kernel_size, stride, padding), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, kernel_size, stride, padding]() {
        if (input->requires_gradient)
            input->gradient = tensor_add(input->gradient,
                max_pool2d_backward(self->gradient, input->value, kernel_size, stride, padding));
    };
    return VarTensor(result);
}

VarTensor VarTensor::avg_pool2d(std::size_t kernel_size, std::size_t stride, std::size_t padding) const {
    if (!node_) throw MatrixProError("Cannot pool an empty tensor variable");
    auto result = make_node(matrix_pro::avg_pool2d(value(), kernel_size, stride, padding), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, kernel_size, stride, padding]() {
        if (input->requires_gradient)
            input->gradient = tensor_add(input->gradient,
                avg_pool2d_backward(self->gradient, input->value, kernel_size, stride, padding));
    };
    return VarTensor(result);
}

VarTensor VarTensor::custom_unary(const std::string& name,
                                  std::function<Tensor(const Tensor&)> forward,
                                  std::function<Tensor(const Tensor&, const Tensor&, const Tensor&)> backward) const {
    if (!node_) throw MatrixProError("Cannot apply a custom op to an empty tensor variable");
    if (!forward || !backward) throw InvalidArgumentError("custom_unary requires forward and backward for " + name);
    auto result = make_node(forward(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, backward]() {
        if (input->requires_gradient)
            input->gradient = tensor_add(input->gradient, backward(self->gradient, input->value, self->value));
    };
    return VarTensor(result);
}

VarTensor custom_loss(const VarTensor& prediction, const VarTensor& target,
                      std::function<float(const Tensor&, const Tensor&)> forward,
                      std::function<Tensor(const Tensor&, const Tensor&)> backward) {
    if (!prediction.node_ || !target.node_) throw MatrixProError("custom_loss requires non-empty tensor variables");
    if (!forward || !backward) throw InvalidArgumentError("custom_loss requires forward and backward");
    const float loss = forward(prediction.value(), target.value());
    auto result = make_node(Tensor({1, 1}, std::vector<float>{loss}),
                            prediction.requires_grad() || target.requires_grad());
    result->parents = {prediction.node_, target.node_};
    result->backward_function = [self = result.get(), pred = prediction.node_, tgt = target.node_, backward]() {
        Tensor upstream = self->gradient;
        upstream.download();   // scalar read; the loss node sits at the graph root
        const float scale = upstream.data()[0];
        if (pred->requires_gradient)
            pred->gradient = tensor_add(pred->gradient, tensor_multiply(backward(pred->value, tgt->value), scale));
        if (tgt->requires_gradient)
            tgt->gradient = tensor_add(tgt->gradient, tensor_multiply(tensor_negate(backward(pred->value, tgt->value)), scale));
    };
    return VarTensor(result);
}

}

#include "matrix_pro/autograd.hpp"
#include "matrix_pro/operations.hpp"

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

namespace {
std::shared_ptr<Variable::Node> make_node(const Matrix& value, bool requires_gradient) {
    auto node = std::make_shared<Variable::Node>();
    node->value = value;
    node->requires_gradient = requires_gradient;
    if (requires_gradient) node->gradient = Matrix::zeros(value.rows(), value.cols());
    return node;
}

void collect_nodes(const std::shared_ptr<Variable::Node>& node,
                   std::unordered_set<Variable::Node*>& visited,
                   std::vector<std::shared_ptr<Variable::Node>>& order) {
    if (!node || !visited.insert(node.get()).second) return;
    for (const auto& parent : node->parents) collect_nodes(parent, visited, order);
    order.push_back(node);
}
}

Variable::Variable(const Matrix& value, bool requires_grad)
    : node_(make_node(value, requires_grad)) {}

Variable::Variable(std::shared_ptr<Node> node) : node_(std::move(node)) {}

const Matrix& Variable::value() const {
    if (!node_) throw std::runtime_error("Empty autograd variable");
    return node_->value;
}

const Matrix& Variable::grad() const {
    if (!node_) throw std::runtime_error("Empty autograd variable");
    return node_->gradient;
}

bool Variable::requires_grad() const { return node_ && node_->requires_gradient; }

void Variable::zero_grad() {
    if (requires_grad()) node_->gradient = Matrix::zeros(value().rows(), value().cols());
}

void Variable::backward() {
    if (!node_) throw std::runtime_error("Cannot backward an empty variable");
    if (!node_->requires_gradient) throw std::invalid_argument("Root variable does not require gradients");
    node_->gradient = Matrix::ones(value().rows(), value().cols());
    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;
    collect_nodes(node_, visited, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_function) (*it)->backward_function();
    }
}

Variable Variable::add(const Variable& other) const {
    if (!node_ || !other.node_) throw std::runtime_error("Cannot add empty variables");
    auto result = make_node(matrix_pro::add(value(), other.value()), requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [result, left = node_, right = other.node_]() {
        if (left->requires_gradient) left->gradient += result->gradient;
        if (right->requires_gradient) right->gradient += result->gradient;
    };
    return Variable(result);
}

Variable Variable::matmul(const Variable& other) const {
    if (!node_ || !other.node_) throw std::runtime_error("Cannot multiply empty variables");
    auto result = make_node(matrix_pro::multiply(value(), other.value()), requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [result, left = node_, right = other.node_]() {
        if (left->requires_gradient) left->gradient += matrix_pro::multiply(result->gradient, matrix_pro::transpose(right->value));
        if (right->requires_gradient) right->gradient += matrix_pro::multiply(matrix_pro::transpose(left->value), result->gradient);
    };
    return Variable(result);
}

Variable Variable::relu() const {
    auto result = make_node(value().relu(), requires_grad());
    result->parents = {node_};
    result->backward_function = [result, input = node_]() {
        if (input->requires_gradient) input->gradient += matrix_pro::elementwise_multiply(result->gradient, input->value.greater(0.0f));
    };
    return Variable(result);
}

Variable Variable::leaky_relu(float negative_slope) const {
    auto result = make_node(value().leaky_relu(negative_slope), requires_grad());
    result->parents = {node_};
    result->backward_function = [result, input = node_, negative_slope]() {
        if (input->requires_gradient) {
            Matrix positive = input->value.greater(0.0f);
            Matrix derivative = matrix_pro::where(positive, 1.0f, negative_slope);
            input->gradient += matrix_pro::elementwise_multiply(result->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::gelu() const {
    auto result = make_node(value().gelu(), requires_grad());
    result->parents = {node_};
    result->backward_function = [result, input = node_]() {
        if (input->requires_gradient) {
            const float inverse_sqrt_two_pi = 0.7978845608f;
            Matrix cdf = (input->value + input->value.pow(3.0f) * 0.044715f) * inverse_sqrt_two_pi;
            Matrix tanh_value = cdf.tanh();
            Matrix term = tanh_value.add_scalar(1.0f);
            Matrix derivative = term * 0.5f;
            derivative += matrix_pro::elementwise_multiply(input->value, (Matrix::ones(input->value.rows(), input->value.cols()) - matrix_pro::elementwise_multiply(tanh_value, tanh_value))) * 0.5f;
            derivative = matrix_pro::elementwise_multiply(derivative, Matrix::ones(input->value.rows(), input->value.cols()));
            input->gradient += matrix_pro::elementwise_multiply(result->gradient, derivative);
        }
    };
    return Variable(result);
}

Variable Variable::swish(float beta) const {
    auto result = make_node(value().swish(beta), requires_grad());
    result->parents = {node_};
    result->backward_function = [result, input = node_, beta]() {
        if (input->requires_gradient) {
            Matrix sigmoid = (input->value * beta).sigmoid();
            Matrix derivative = sigmoid + matrix_pro::elementwise_multiply(input->value * beta,
                matrix_pro::elementwise_multiply(sigmoid, Matrix::ones(input->value.rows(), input->value.cols()) - sigmoid));
            input->gradient += matrix_pro::elementwise_multiply(result->gradient, derivative);
        }
    };
    return Variable(result);
}

}

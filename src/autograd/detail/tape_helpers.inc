// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Düğüm kurma ve topolojik sıralama; gradyanlar sıfırdan biriktirilir.

namespace {

std::shared_ptr<Variable::Node> make_node(const Matrix& value, bool requires_gradient) {
    requires_gradient = requires_gradient && is_grad_enabled();
    auto node = std::make_shared<Variable::Node>();
    node->value = value;
    node->requires_gradient = requires_gradient;
    if (requires_gradient)
        node->gradient = Matrix::zeros(value.rows(), value.cols());
    return node;
}

std::shared_ptr<VarTensor::Node> make_node(const Tensor& value, bool requires_gradient) {
    requires_gradient = requires_gradient && is_grad_enabled();
    auto node = std::make_shared<VarTensor::Node>();
    node->value = value;
    node->requires_gradient = requires_gradient;
    if (requires_gradient) {
        node->gradient = Tensor(value.shape());
        node->gradient.zero(); // accumulators, never garbage
    }
    return node;
}

void collect_nodes(const std::shared_ptr<Variable::Node>& node,
                   std::unordered_set<Variable::Node*>& visited,
                   std::vector<std::shared_ptr<Variable::Node>>& order) {
    if (!node || !visited.insert(node.get()).second)
        return;
    for (const auto& parent : node->parents)
        collect_nodes(parent, visited, order);
    order.push_back(node);
}

void collect_nodes(const std::shared_ptr<VarTensor::Node>& node,
                   std::unordered_set<VarTensor::Node*>& visited,
                   std::vector<std::shared_ptr<VarTensor::Node>>& order) {
    if (!node || !visited.insert(node.get()).second)
        return;
    for (const auto& parent : node->parents)
        collect_nodes(parent, visited, order);
    order.push_back(node);
}

Matrix scalar_matrix(float value) {
    return Matrix(1, 1, std::vector<float>{value});
}

Matrix band_mask(const Matrix& m, float low, float high) {
    return matrix_pro::elementwise_multiply(m.greater(low), m.less(high));
}

}

// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Variable yaşam döngüsü, sum ve ters topolojik backward.

Variable::Variable(const Matrix& value, bool requires_grad)
    : node_(make_node(value, requires_grad)) {
}

Variable::Variable(std::shared_ptr<Node> node) : node_(std::move(node)) {
}

const Matrix& Variable::value() const {
    if (!node_)
        throw MatrixProError("Empty autograd variable");
    return node_->value;
}

const Matrix& Variable::grad() const {
    if (!node_)
        throw MatrixProError("Empty autograd variable");
    return node_->gradient;
}

bool Variable::requires_grad() const {
    return node_ && node_->requires_gradient;
}

Variable Variable::sum() const {
    auto result = make_node(scalar_matrix(value().sum()), requires_grad());
    result->parents = {node_};
    auto parent = node_;
    auto* self = result.get();
    result->backward_function = [parent, self] {
        if (parent->requires_gradient) {
            Matrix zeros(parent->value.rows(), parent->value.cols(), MemoryMode::device_only);
            zeros.fill(0);
            parent->gradient += matrix_pro::broadcast_add(zeros, self->gradient);
        }
    };
    return Variable(result);
}

void Variable::zero_grad() {
    if (requires_grad())
        node_->gradient = Matrix::zeros(value().rows(), value().cols());
}

void Variable::backward() {
    if (!node_)
        throw MatrixProError("Cannot backward an empty variable");
    if (!node_->requires_gradient)
        throw InvalidArgumentError("Root variable does not require gradients");
    node_->gradient = Matrix::ones(value().rows(), value().cols());
    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;
    collect_nodes(node_, visited, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_function)
            (*it)->backward_function();
    }
}

// --- arithmetic ---

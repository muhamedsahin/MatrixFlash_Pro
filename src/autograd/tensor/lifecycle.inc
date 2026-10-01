// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Rank-N tape erişimi ve backward başlatma.

VarTensor::VarTensor(const Tensor& value, bool requires_grad)
    : node_(make_node(value, requires_grad)) {
}

VarTensor::VarTensor(std::shared_ptr<Node> node) : node_(std::move(node)) {
}

const Tensor& VarTensor::value() const {
    if (!node_)
        throw MatrixProError("Empty autograd tensor");
    return node_->value;
}

const Tensor& VarTensor::grad() const {
    if (!node_)
        throw MatrixProError("Empty autograd tensor");
    return node_->gradient;
}

bool VarTensor::requires_grad() const {
    return node_ && node_->requires_gradient;
}

void VarTensor::zero_grad() {
    if (requires_grad())
        node_->gradient.zero();
}

void VarTensor::backward() {
    if (!node_)
        throw MatrixProError("Cannot backward an empty tensor variable");
    if (!node_->requires_gradient)
        throw InvalidArgumentError("Root tensor variable does not require gradients");
    node_->gradient = Tensor(value().shape());
    node_->gradient.fill(1.0f);
    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;
    collect_nodes(node_, visited, order);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_function)
            (*it)->backward_function();
    }
}

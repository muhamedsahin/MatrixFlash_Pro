// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Tensor için kullanıcı tanımlı ileri/geri işlemler.

VarTensor VarTensor::custom_unary(
    const std::string& name, std::function<Tensor(const Tensor&)> forward,
    std::function<Tensor(const Tensor&, const Tensor&, const Tensor&)> backward) const {
    if (!node_)
        throw MatrixProError("Cannot apply a custom op to an empty tensor variable");
    if (!forward || !backward)
        throw InvalidArgumentError("custom_unary requires forward and backward for " + name);
    auto result = make_node(forward(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, backward]() {
        if (input->requires_gradient)
            input->gradient =
                tensor_add(input->gradient, backward(self->gradient, input->value, self->value));
    };
    return VarTensor(result);
}

VarTensor custom_loss(const VarTensor& prediction, const VarTensor& target,
                      std::function<float(const Tensor&, const Tensor&)> forward,
                      std::function<Tensor(const Tensor&, const Tensor&)> backward) {
    if (!prediction.node_ || !target.node_)
        throw MatrixProError("custom_loss requires non-empty tensor variables");
    if (!forward || !backward)
        throw InvalidArgumentError("custom_loss requires forward and backward");
    const float loss = forward(prediction.value(), target.value());
    auto result = make_node(Tensor({1, 1}, std::vector<float>{loss}),
                            prediction.requires_grad() || target.requires_grad());
    result->parents = {prediction.node_, target.node_};
    result->backward_function = [self = result.get(), pred = prediction.node_, tgt = target.node_,
                                 backward]() {
        Tensor upstream = self->gradient;
        upstream.download(); // scalar read; the loss node sits at the graph root
        const float scale = upstream.data()[0];
        if (pred->requires_gradient)
            pred->gradient = tensor_add(pred->gradient,
                                        tensor_multiply(backward(pred->value, tgt->value), scale));
        if (tgt->requires_gradient)
            tgt->gradient = tensor_add(
                tgt->gradient,
                tensor_multiply(tensor_negate(backward(pred->value, tgt->value)), scale));
    };
    return VarTensor(result);
}

// Thread-local gradient tracking state

// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Tensor aritmetiği ve batched GEMM türevleri.

VarTensor VarTensor::add(const VarTensor& other) const {
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot add empty tensor variables");
    auto result =
        make_node(tensor_add(value(), other.value()), requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient = tensor_add(left->gradient, self->gradient);
        if (right->requires_gradient)
            right->gradient = tensor_add(right->gradient, self->gradient);
    };
    return VarTensor(result);
}

VarTensor VarTensor::elementwise_multiply(const VarTensor& other) const {
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot multiply empty tensor variables");
    auto result = make_node(tensor_multiply(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient =
                tensor_add(left->gradient, tensor_multiply(self->gradient, right->value));
        if (right->requires_gradient)
            right->gradient =
                tensor_add(right->gradient, tensor_multiply(self->gradient, left->value));
    };
    return VarTensor(result);
}

VarTensor VarTensor::multiply(float scalar) const {
    if (!node_)
        throw MatrixProError("Cannot scale an empty tensor variable");
    auto result = make_node(tensor_multiply(value(), scalar), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, scalar]() {
        if (input->requires_gradient)
            input->gradient = tensor_add(input->gradient, tensor_multiply(self->gradient, scalar));
    };
    return VarTensor(result);
}

VarTensor VarTensor::negate() const {
    return multiply(-1.0f);
}

VarTensor VarTensor::batch_matmul(const VarTensor& other) const {
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot batch-multiply empty tensor variables");
    auto result = make_node(matrix_pro::batch_matmul(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient = tensor_add(left->gradient,
                                        batch_matmul_backward_left(self->gradient, right->value));
        if (right->requires_gradient)
            right->gradient = tensor_add(right->gradient,
                                         batch_matmul_backward_right(self->gradient, left->value));
    };
    return VarTensor(result);
}

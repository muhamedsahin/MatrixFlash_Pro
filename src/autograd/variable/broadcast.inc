// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Broadcast türevleri genişletilen eksenlerde geri toplanır.

Variable Variable::add_row_vector(const Variable& vector) const {
    if (!node_ || !vector.node_)
        throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::add_row_vector(value(), vector.value()),
                            requires_grad() || vector.requires_grad());
    result->parents = {node_, vector.node_};
    result->backward_function = [self = result.get(), input = node_, vector = vector.node_]() {
        if (input->requires_gradient)
            input->gradient += self->gradient;
        if (vector->requires_gradient)
            vector->gradient += matrix_pro::broadcast_backward(self->gradient, vector->value);
    };
    return Variable(result);
}

Variable Variable::add_col_vector(const Variable& vector) const {
    if (!node_ || !vector.node_)
        throw MatrixProError("Cannot broadcast empty variables");
    auto result = make_node(matrix_pro::add_col_vector(value(), vector.value()),
                            requires_grad() || vector.requires_grad());
    result->parents = {node_, vector.node_};
    result->backward_function = [self = result.get(), input = node_, vector = vector.node_]() {
        if (input->requires_gradient)
            input->gradient += self->gradient;
        if (vector->requires_gradient)
            vector->gradient += matrix_pro::broadcast_backward(self->gradient, vector->value);
    };
    return Variable(result);
}

Variable Variable::multiply_row_vector(const Variable& vector) const {
    if (!node_ || !vector.node_)
        throw MatrixProError("Cannot broadcast empty variables");
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
    if (!node_ || !vector.node_)
        throw MatrixProError("Cannot broadcast empty variables");
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
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot broadcast empty variables");
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
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot broadcast empty variables");
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

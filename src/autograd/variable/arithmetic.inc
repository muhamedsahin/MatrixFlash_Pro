// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matris ve skaler aritmetiğin ileri/geri türevleri.

Variable Variable::add(const Variable& other) const {
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot add empty variables");
    auto result = make_node(matrix_pro::add(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient += self->gradient;
        if (right->requires_gradient)
            right->gradient += self->gradient;
    };
    return Variable(result);
}

Variable Variable::subtract(const Variable& other) const {
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot subtract empty variables");
    auto result = make_node(matrix_pro::subtract(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient += self->gradient;
        if (right->requires_gradient)
            right->gradient -= self->gradient;
    };
    return Variable(result);
}

Variable Variable::elementwise_multiply(const Variable& other) const {
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot multiply empty variables");
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
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot divide empty variables");
    auto result = make_node(matrix_pro::divide(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        // d/dA = g / B ;  d/dB = -g * A / B^2
        if (left->requires_gradient)
            left->gradient += matrix_pro::divide(self->gradient, right->value);
        if (right->requires_gradient) {
            Matrix quotient =
                matrix_pro::divide(matrix_pro::elementwise_multiply(self->gradient, left->value),
                                   matrix_pro::elementwise_multiply(right->value, right->value));
            right->gradient -= quotient;
        }
    };
    return Variable(result);
}

Variable Variable::matmul(const Variable& other) const {
    if (!node_ || !other.node_)
        throw MatrixProError("Cannot multiply empty variables");
    auto result = make_node(matrix_pro::multiply(value(), other.value()),
                            requires_grad() || other.requires_grad());
    result->parents = {node_, other.node_};
    result->backward_function = [self = result.get(), left = node_, right = other.node_]() {
        if (left->requires_gradient)
            left->gradient +=
                matrix_pro::multiply(self->gradient, matrix_pro::transpose(right->value));
        if (right->requires_gradient)
            right->gradient +=
                matrix_pro::multiply(matrix_pro::transpose(left->value), self->gradient);
    };
    return Variable(result);
}

Variable Variable::multiply(float scalar) const {
    if (!node_)
        throw MatrixProError("Cannot scale an empty variable");
    auto result = make_node(value() * scalar, requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, scalar]() {
        if (input->requires_gradient)
            input->gradient += self->gradient * scalar;
    };
    return Variable(result);
}

Variable Variable::add_scalar(float scalar) const {
    if (!node_)
        throw MatrixProError("Cannot shift an empty variable");
    auto result = make_node(value().add_scalar(scalar), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient)
            input->gradient += self->gradient;
    };
    return Variable(result);
}

Variable Variable::negate() const {
    return multiply(-1.0f);
}

Variable Variable::pow(float exponent) const {
    if (!node_)
        throw MatrixProError("Cannot exponentiate an empty variable");
    auto result = make_node(value().pow(exponent), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, exponent]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::elementwise_multiply(
                self->gradient, input->value.pow(exponent - 1.0f) * exponent);
    };
    return Variable(result);
}

// --- broadcast (bias-style) ops ---

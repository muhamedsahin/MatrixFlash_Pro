// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Aktivasyon türevleri; mevcut GPU primitifleri ve formüller korunur.

Variable Variable::relu() const {
    auto result = make_node(value().relu(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient)
            input->gradient +=
                matrix_pro::elementwise_multiply(self->gradient, input->value.greater(0.0f));
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
            Matrix derivative =
                matrix_pro::elementwise_multiply(s, Matrix::ones(s.rows(), s.cols()) - s);
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
            Matrix derivative =
                Matrix::ones(t.rows(), t.cols()) - matrix_pro::elementwise_multiply(t, t);
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
            Matrix du_dx = (Matrix::ones(input->value.rows(), input->value.cols()) +
                            input->value.pow(2.0f) * (3.0f * 0.044715f)) *
                           inverse_sqrt_two_pi;
            derivative +=
                matrix_pro::elementwise_multiply(
                    input->value, matrix_pro::elementwise_multiply(
                                      Matrix::ones(input->value.rows(), input->value.cols()) -
                                          matrix_pro::elementwise_multiply(tanh_value, tanh_value),
                                      du_dx)) *
                0.5f;
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
            Matrix derivative =
                sigmoid +
                matrix_pro::elementwise_multiply(
                    input->value * beta,
                    matrix_pro::elementwise_multiply(
                        sigmoid, Matrix::ones(input->value.rows(), input->value.cols()) - sigmoid));
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
            input->gradient += matrix_pro::elementwise_multiply(self->gradient,
                                                                band_mask(input->value, low, high));
    };
    return Variable(result);
}

Variable Variable::hardsigmoid() const {
    auto result = make_node(matrix_pro::hardsigmoid(value()), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::elementwise_multiply(
                self->gradient, band_mask(input->value, -3.0f, 3.0f) * (1.0f / 6.0f));
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
            Matrix inner = matrix_pro::elementwise_multiply(band_mask(x, -3.0f, 3.0f),
                                                            (x * (1.0f / 3.0f)).add_scalar(0.5f));
            Matrix derivative =
                matrix_pro::where(x.greater(3.0f), Matrix::ones(x.rows(), x.cols()), inner);
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
    if (!node_ || !alpha.node_)
        throw MatrixProError("Cannot apply PReLU to empty variables");
    auto result = make_node(matrix_pro::prelu(value(), alpha.value()),
                            requires_grad() || alpha.requires_grad());
    result->parents = {node_, alpha.node_};
    result->backward_function = [self = result.get(), input = node_, alpha = alpha.node_]() {
        if (input->requires_gradient)
            input->gradient +=
                matrix_pro::where(input->value.greater(0.0f), self->gradient,
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
                if (contribution.rows() == alpha->value.rows() &&
                    contribution.cols() == alpha->value.cols())
                    alpha->gradient += contribution;
                else
                    throw ShapeMismatchError("PReLU backward: alpha gradient shape mismatch");
            }
        }
    };
    return Variable(result);
}

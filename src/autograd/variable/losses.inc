// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Özel işlemler ve kayıplar; upstream gradyan ölçeklemesi burada yapılır.

Variable Variable::custom_unary(
    const std::string& name, std::function<Matrix(const Matrix&)> forward,
    std::function<Matrix(const Matrix&, const Matrix&, const Matrix&)> backward) const {
    if (!node_)
        throw MatrixProError("Cannot apply a custom op to an empty variable");
    if (!forward || !backward)
        throw InvalidArgumentError("custom_unary requires forward and backward for " + name);
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
    if (!prediction.node_ || !target.node_)
        throw MatrixProError("custom_loss requires non-empty variables");
    if (!forward || !backward)
        throw InvalidArgumentError("custom_loss requires forward and backward");
    auto result = make_node(scalar_matrix(forward(prediction.value(), target.value())),
                            prediction.requires_grad() || target.requires_grad());
    result->parents = {prediction.node_, target.node_};
    result->backward_function = [self = result.get(), pred = prediction.node_, tgt = target.node_,
                                 backward]() {
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
    return custom_loss(
        prediction, target,
        [](const Matrix& p, const Matrix& t) {
            Matrix diff = matrix_pro::subtract(p, t);
            return sum(matrix_pro::elementwise_multiply(diff, diff)) / static_cast<float>(p.size());
        },
        [](const Matrix& p, const Matrix& t) {
            Matrix diff = matrix_pro::subtract(p, t);
            return (matrix_pro::elementwise_multiply(diff,
                                                     Matrix::ones(p.rows(), p.cols()) * 2.0f)) *
                   (1.0f / static_cast<float>(p.size()));
        });
}

Variable mae_loss(const Variable& prediction, const Variable& target) {
    return custom_loss(
        prediction, target,
        [](const Matrix& p, const Matrix& t) {
            return sum(matrix_pro::abs(matrix_pro::subtract(p, t))) / static_cast<float>(p.size());
        },
        [](const Matrix& p, const Matrix& t) {
            // sign(p - t) with subgradient 0 where p == t.
            Matrix diff = matrix_pro::subtract(p, t);
            Matrix ones = Matrix::ones(p.rows(), p.cols());
            Matrix sign =
                matrix_pro::where(diff.greater(0.0f), ones,
                                  matrix_pro::where(diff.less(0.0f), matrix_pro::negate(ones),
                                                    Matrix::zeros(p.rows(), p.cols())));
            return sign * (1.0f / static_cast<float>(p.size()));
        });
}

Variable huber_loss(const Variable& prediction, const Variable& target, float delta) {
    return custom_loss(
        prediction, target,
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
            Matrix sign =
                matrix_pro::where(diff.greater(0.0f), ones,
                                  matrix_pro::where(diff.less(0.0f), matrix_pro::negate(ones),
                                                    Matrix::zeros(p.rows(), p.cols())));
            Matrix grad = matrix_pro::where(abs_diff.greater(delta), sign * delta, diff);
            return grad * (1.0f / static_cast<float>(p.size()));
        });
}

Variable bce_with_logits_loss(const Variable& logits, const Variable& target) {
    return custom_loss(
        logits, target,
        [](const Matrix& z, const Matrix& t) { return bce_with_logits_value(z, t); },
        [](const Matrix& z, const Matrix& t) { return bce_with_logits_grad(z, t); });
}

Variable softmax_cross_entropy_loss(const Variable& logits, const Variable& one_hot_target) {
    return custom_loss(
        logits, one_hot_target,
        [](const Matrix& z, const Matrix& t) {
            return sum(softmax_cross_entropy_value(z, t)) / static_cast<float>(z.rows());
        },
        [](const Matrix& z, const Matrix& t) { return softmax_cross_entropy_grad(z, t); });
}

Variable kl_divergence_loss(const Variable& prediction, const Variable& target) {
    return custom_loss(
        prediction, target,
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

// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Concat, reshape ve slice; geri geçiş orijinal şekle dağıtılır.

Variable Variable::concat(const std::vector<Variable>& parts, std::size_t axis) {
    if (parts.empty())
        throw InvalidArgumentError("concat requires at least one variable");
    if (parts.size() == 1)
        return parts.front();
    std::vector<Matrix> values;
    std::vector<std::shared_ptr<Node>> parents;
    values.reserve(parts.size());
    parents.reserve(parts.size());
    bool
        requires
    = false;
    for (const Variable& part : parts) {
        if (!part.node_)
            throw MatrixProError("Cannot concat empty variables");
        values.push_back(part.value());
        parents.push_back(part.node_);
        requires =
                     requires || part
                     .requires_grad();
    }
    auto result = make_node(matrix_pro::concat(values, axis), requires);
    result->parents = parents;
    result->backward_function = [self = result.get(), parents, axis]() {
        std::size_t offset = 0;
        for (const auto& parent : parents) {
            const std::size_t rows = parent->value.rows();
            const std::size_t cols = parent->value.cols();
            if (parent->requires_gradient) {
                // Each operand simply receives the slice of the upstream gradient
                // that it contributed during the forward pass.
                parent->gradient +=
                    axis == 0 ? matrix_pro::slice(self->gradient, offset, offset + rows, 0, cols)
                              : matrix_pro::slice(self->gradient, 0, rows, offset, offset + cols);
            }
            offset += axis == 0 ? rows : cols;
        }
    };
    return Variable(result);
}

// --- transforms ---

Variable Variable::transpose() const {
    if (!node_)
        throw MatrixProError("Cannot transpose an empty variable");
    auto result = make_node(value().transpose(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_]() {
        if (input->requires_gradient)
            input->gradient += self->gradient.transpose();
    };
    return Variable(result);
}

Variable Variable::flatten() const {
    if (!node_)
        throw MatrixProError("Cannot flatten an empty variable");
    const std::size_t rows = value().rows(), cols = value().cols();
    auto result = make_node(value().flatten(), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, rows, cols]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::reshape(self->gradient, rows, cols);
    };
    return Variable(result);
}

Variable Variable::reshape(std::size_t rows, std::size_t cols) const {
    if (!node_)
        throw ShapeMismatchError("Cannot reshape an empty variable");
    const std::size_t original_rows = value().rows(), original_cols = value().cols();
    auto result = make_node(matrix_pro::reshape(value(), rows, cols), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, original_rows,
                                 original_cols]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::reshape(self->gradient, original_rows, original_cols);
    };
    return Variable(result);
}

Variable Variable::slice(std::size_t row_start, std::size_t row_end, std::size_t col_start,
                         std::size_t col_end) const {
    if (!node_)
        throw OutOfRangeError("Cannot slice an empty variable");
    auto result = make_node(value().slice(row_start, row_end, col_start, col_end), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, row_start, row_end, col_start,
                                 col_end]() {
        if (input->requires_gradient)
            input->gradient += matrix_pro::slice_scatter(self->gradient, input->value, row_start,
                                                         row_end, col_start, col_end);
    };
    return Variable(result);
}

// --- activations ---

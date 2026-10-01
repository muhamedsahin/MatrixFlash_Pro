// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Detach, clone, gradient clipping ve checkpoint yeniden hesaplama.

Variable Variable::detach() const {
    Variable result(value(), false);
    return result;
}

Variable Variable::clone() const {
    Variable result{Matrix(value())};
    return result;
}

bool Variable::is_leaf() const {
    return node_->parents.empty();
}

void Variable::retain_grad() {
    node_->retain_grad = true;
}

void Variable::clip_grad_norm_(std::vector<Variable>& params, float max_norm, float norm_type) {
    double total_norm = 0.0;
    for (auto& p : params) {
        if (!p.requires_grad())
            continue;
        const auto& g = p.grad();
        if (norm_type == 2.0f) {
            float gn = g.l2_norm();
            total_norm += static_cast<double>(gn) * gn;
        } else if (norm_type == 1.0f) {
            total_norm += g.l1_norm();
        } else {
            total_norm = std::max(total_norm, static_cast<double>(g.abs_max()));
        }
    }
    if (norm_type == 2.0f)
        total_norm = std::sqrt(total_norm);

    float clip_coef = max_norm / (static_cast<float>(total_norm) + 1e-6f);
    if (clip_coef >= 1.0f)
        return;

    for (auto& p : params) {
        if (!p.requires_grad())
            continue;
        p.node_->gradient = p.node_->gradient * clip_coef;
    }
}

void Variable::clip_grad_value_(std::vector<Variable>& params, float clip_value) {
    for (auto& p : params) {
        if (!p.requires_grad())
            continue;
        p.node_->gradient = clamp(p.node_->gradient, -clip_value, clip_value);
    }
}

Variable Variable::checkpoint(std::function<Variable(const Variable&)> fn, const Variable& input) {
    Variable output;
    {
        NoGradGuard no_grad;
        output = fn(input);
    }

    auto result = Variable(output.value(), true);
    if (input.requires_grad()) {
        auto fn_copy = fn;
        auto input_node = input.node_;
        auto result_node = result.node_;
        result_node->parents = {input_node};
        result_node->backward_function = [fn_copy, input_node, result_node]() {
            Variable recomputed_input(input_node->value, true);
            Variable recomputed_output = fn_copy(recomputed_input);

            recomputed_output.elementwise_multiply(Variable(result_node->gradient, false))
                .sum()
                .backward();

            input_node->gradient = matrix_pro::add(input_node->gradient, recomputed_input.grad());
        };
    }
    return result;
}

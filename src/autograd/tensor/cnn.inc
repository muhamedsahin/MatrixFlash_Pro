// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: NCHW convolution ve pooling gradyanları.

VarTensor VarTensor::conv2d(const VarTensor& weights, const VarTensor& bias, std::size_t stride,
                            std::size_t padding) const {
    if (!node_ || !weights.node_ || !bias.node_)
        throw MatrixProError("conv2d requires non-empty tensor variables");
    auto result =
        make_node(matrix_pro::conv2d(value(), weights.value(), bias.value(), stride, padding),
                  requires_grad() || weights.requires_grad() || bias.requires_grad());
    result->parents = {node_, weights.node_, bias.node_};
    result->backward_function = [self = result.get(), input = node_, weight = weights.node_,
                                 bias = bias.node_, stride, padding]() {
        if (input->requires_gradient)
            input->gradient =
                tensor_add(input->gradient, conv2d_input_backward(self->gradient, input->value,
                                                                  weight->value, stride, padding));
        if (weight->requires_gradient)
            weight->gradient = tensor_add(weight->gradient,
                                          conv2d_weight_backward(self->gradient, input->value,
                                                                 weight->value, stride, padding));
        if (bias->requires_gradient)
            bias->gradient = tensor_add(bias->gradient, conv2d_bias_backward(self->gradient));
    };
    return VarTensor(result);
}

VarTensor VarTensor::max_pool2d(std::size_t kernel_size, std::size_t stride,
                                std::size_t padding) const {
    if (!node_)
        throw MatrixProError("Cannot pool an empty tensor variable");
    auto result =
        make_node(matrix_pro::max_pool2d(value(), kernel_size, stride, padding), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, kernel_size, stride,
                                 padding]() {
        if (input->requires_gradient)
            input->gradient =
                tensor_add(input->gradient, max_pool2d_backward(self->gradient, input->value,
                                                                kernel_size, stride, padding));
    };
    return VarTensor(result);
}

VarTensor VarTensor::avg_pool2d(std::size_t kernel_size, std::size_t stride,
                                std::size_t padding) const {
    if (!node_)
        throw MatrixProError("Cannot pool an empty tensor variable");
    auto result =
        make_node(matrix_pro::avg_pool2d(value(), kernel_size, stride, padding), requires_grad());
    result->parents = {node_};
    result->backward_function = [self = result.get(), input = node_, kernel_size, stride,
                                 padding]() {
        if (input->requires_gradient)
            input->gradient =
                tensor_add(input->gradient, avg_pool2d_backward(self->gradient, input->value,
                                                                kernel_size, stride, padding));
    };
    return VarTensor(result);
}

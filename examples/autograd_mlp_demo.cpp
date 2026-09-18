// ============================================================================
//  MatrixFlash-Pro — autograd MLP demo
//  ---------------------------------------------------------------------------
//  Trains a two-layer MLP on a synthetic classification task with the
//  reverse-mode engine from matrix_pro/autograd/autograd.hpp:
//
//      hidden = relu(x @ W1 + b1)
//      logits = hidden @ W2 + b2
//      loss   = softmax_cross_entropy(logits, one_hot_target)
//      loss.backward()   ->   .grad() of every participating Variable
//
//  Everything (forward, fused loss, backward) runs on the GPU; the host only
//  observes the printed loss.
//
//  Build & run:
//     cmake --build --preset release --target matrix_pro_demo_mlp
//     ./build/Release/matrix_pro_demo_mlp
// ============================================================================

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "matrix_pro/matrix_pro.hpp"

namespace {

using matrix_pro::Matrix;
using matrix_pro::Variable;

// Deterministic synthetic classification task: the label of a sample is the
// index of its largest input value.
Matrix build_inputs(std::size_t samples, std::size_t features, std::uint32_t seed) {
	return Matrix::uniform(samples, features, 0.0f, 1.0f, seed);
}

Matrix build_labels(const Matrix& inputs, std::size_t classes) {
	const std::vector<float>& values = inputs.data();
	std::vector<float> labels(inputs.rows());
	for (std::size_t row = 0; row < inputs.rows(); ++row) {
		std::size_t best = 0;
		for (std::size_t col = 1; col < inputs.cols(); ++col) {
			if (values[row * inputs.cols() + col] > values[row * inputs.cols() + best]) {
				best = col;
			}
		}
		labels[row] = static_cast<float>(best % classes);
	}
	return matrix_pro::one_hot(Matrix(inputs.rows(), 1, labels), classes);
}

} // namespace

int main() {
	const std::size_t samples = 256;
	const std::size_t features = 64;
	const std::size_t hidden = 128;
	const std::size_t classes = 10;
	const int steps = 60;
	const float learning_rate = 0.5f;

	const Matrix inputs_matrix = build_inputs(samples, features, 20260917u);
	const Matrix labels_matrix = build_labels(inputs_matrix, classes);

	Variable inputs(inputs_matrix, false);
	Variable targets(labels_matrix, false);

	// Parameters: glorot for the weights, zeros for the biases. The shapes follow
	// the (rows x features) * (features x units) convention of Variable::matmul.
	Variable first_weights(Matrix::glorot(features, hidden));
	Variable first_bias(Matrix::zeros(1, hidden));
	Variable second_weights(Matrix::glorot(hidden, classes));
	Variable second_bias(Matrix::zeros(1, classes));

	std::cout << "MatrixFlash-Pro MLP demo: " << samples << " samples, " << features
	          << " features, " << hidden << " hidden units, " << classes << " classes\n";
	std::cout << std::fixed << std::setprecision(4);

	float first_loss = 0.0f;
	float last_loss = 0.0f;

	for (int step = 1; step <= steps; ++step) {
		first_weights.zero_grad();
		first_bias.zero_grad();
		second_weights.zero_grad();
		second_bias.zero_grad();

		Variable hidden_layer = inputs.matmul(first_weights).broadcast_add(first_bias).relu();
		Variable logits = hidden_layer.matmul(second_weights).broadcast_add(second_bias);
		Variable loss = matrix_pro::softmax_cross_entropy_loss(logits, targets);
		loss.backward();

		// Plain SGD step: p -= lr * grad  (grad() is a Matrix view, so make a copy
		// before materializing it on the host).
		const auto descend = [learning_rate](Variable& parameter) {
			Matrix gradient = parameter.grad();
			gradient.download();
			parameter = Variable(parameter.value() - gradient * learning_rate,
			                     parameter.requires_grad());
		};
		descend(first_weights);
		descend(first_bias);
		descend(second_weights);
		descend(second_bias);

		Matrix loss_value = loss.value();
		loss_value.download();
		const float current = loss_value.at(0, 0);
		if (step == 1) {
			first_loss = current;
		}
		last_loss = current;

		if (step == 1 || step % 15 == 0) {
			std::cout << "  step " << std::setw(3) << step << "  loss = " << current << "\n";
		}
	}

	std::cout << "loss: " << first_loss << " -> " << last_loss << "\n";
	return last_loss < first_loss ? 0 : 1;
}

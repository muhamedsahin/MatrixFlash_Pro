// ============================================================================
//  MatrixFlash-Pro — high performance showcase
//  ---------------------------------------------------------------------------
//  Walks through the features that make the library "fast by default":
//
//    1. device-only matrices  : no host mirror, no hidden PCIe traffic
//    2. cuBLAS GEMM loop      : sustained throughput on resident buffers
//    3. fused / in-place ops  : one kernel instead of a chain of temporaries
//    4. zero-copy views       : transpose / slice without touching memory
//    5. async reductions      : stream-pool argmax into pinned host memory
//
//  Build & run:
//     cmake --build --preset release --target matrix_pro_demo_high_perf
//     ./build/Release/matrix_pro_demo_high_perf
// ============================================================================

#include <chrono>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "matrix_pro/matrix_pro.hpp"

namespace {

using matrix_pro::Matrix;

// Host-side wall clock; the device is synchronized around every measurement so
// the figure includes the actual kernel execution time.
template <typename Fn>
double time_ms(Fn&& body, int repetitions) {
	matrix_pro::synchronize();
	const auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < repetitions; ++i) {
		body();
	}
	matrix_pro::synchronize();
	const auto stop = std::chrono::high_resolution_clock::now();
	const double total = std::chrono::duration<double, std::milli>(stop - start).count();
	return total / static_cast<double>(repetitions);
}

void print_header(const char* title) {
	const std::size_t title_length = std::strlen(title);
	std::cout << "\n=== " << title << " "
	          << std::string(title_length < 56 ? 56 - title_length : 0, '=') << "\n";
}

} // namespace

int main() {
	using matrix_pro::MemoryMode;

	const std::size_t size = 512;
	std::cout << "MatrixFlash-Pro high performance showcase (device "
	          << matrix_pro::current_device() << ")\n";

	// --- 1. device-only matrices --------------------------------------------
	print_header("device-only matrices");
	{
		Matrix resident(size, size, MemoryMode::device_only);
		resident.fill(0.5f);
		const auto start = std::chrono::high_resolution_clock::now();
		resident.download();
		const auto stop = std::chrono::high_resolution_clock::now();
		const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
		std::cout << std::fixed << std::setprecision(3)
		          << "  " << size << "x" << size << " fill + materialization: " << ms
		          << " ms (no host mirror existed before download())\n";
	}

	// --- 2. sustained cuBLAS throughput -------------------------------------
	print_header("cuBLAS GEMM throughput");
	{
		Matrix left = Matrix::random(size, size);
		Matrix right = Matrix::random(size, size);
		const int repetitions = 8;
		const double ms = time_ms([&left, &right] {
			Matrix product = left * right;
			(void)product;
		}, repetitions);
		const double flops = 2.0 * static_cast<double>(size) * size * size;
		std::cout << std::fixed << std::setprecision(3)
		          << "  " << size << "x" << size << " GEMM: " << ms << " ms ("
		          << std::setprecision(1) << flops / (ms * 1.0e6) << " GFLOPS, "
		          << repetitions << " runs averaged)\n";
	}

	// --- 3. fused and in-place kernels --------------------------------------
	print_header("fused / in-place kernels");
	{
		Matrix value = Matrix::uniform(size, 64, -1.0f, 1.0f);
		Matrix gate = Matrix::uniform(size, 64, 0.0f, 1.0f);
		// fused_* chains are strictly elementwise: the bias must match the shape
		// of the input (use broadcast_add first for per-column biases).
		Matrix bias = Matrix::zeros(size, 64);

		const double chain_ms = time_ms([&] {
			Matrix activated = matrix_pro::sigmoid(value);
			Matrix result = matrix_pro::elementwise_multiply(activated, gate);
			(void)result;
		}, 20);

		const double fused_ms = time_ms([&] {
			Matrix result = matrix_pro::fused_sigmoid_mul(value, gate);
			(void)result;
		}, 20);

		const double gelu_ms = time_ms([&] {
			Matrix result = matrix_pro::fused_bias_gelu(value, bias);
			(void)result;
		}, 20);

		std::cout << std::fixed << std::setprecision(4)
		          << "  sigmoid(x)*y : chain " << chain_ms << " ms vs fused " << fused_ms
		          << " ms\n";
		std::cout << "  gelu(x+bias) : fused " << gelu_ms << " ms\n";

		Matrix accumulator = Matrix::ones(size, 64);
		const double inplace_ms = time_ms([&] { matrix_pro::relu_(accumulator); }, 20);
		std::cout << "  relu_        : " << inplace_ms
		          << " ms (writes back into the same buffer)\n";
	}

	// --- 4. zero-copy views --------------------------------------------------
	print_header("zero-copy views");
	{
		Matrix owner = Matrix::random(256, 128);
		matrix_pro::MatrixView transposed = matrix_pro::transpose_view(owner);
		matrix_pro::MatrixView patch = matrix_pro::slice_view(owner, 0, 64, 0, 64);
		Matrix materialized = matrix_pro::materialize(transposed);
		std::cout << "  transpose_view : " << transposed.rows() << "x" << transposed.cols()
		          << " (no copy), materialize -> " << materialized.rows() << "x"
		          << materialized.cols() << "\n";
		std::cout << "  slice_view     : " << patch.rows() << "x" << patch.cols()
		          << " strided window over the same buffer\n";
	}

	// --- 5. asynchronous reduction on the stream pool ------------------------
	print_header("async reductions");
	{
		Matrix scores = Matrix::random(1, 4096);
		std::size_t* slot = static_cast<std::size_t*>(matrix_pro::pin_host(sizeof(std::size_t)));
		*slot = 0;

		matrix_pro::argmax_async(scores, slot);
		matrix_pro::synchronize_pool();
		std::cout << "  argmax_async on the pool stream -> index " << *slot
		          << " (pinned host slot, no pipeline stall)\n";
		matrix_pro::unpin_host(slot);
	}

	std::cout << "\ndone.\n";
	return 0;
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "matrix_pro/core/memory_mode.hpp"

namespace matrix_pro {

// Forward declarations for decomposition results (defined in operations.hpp).
struct QRResult;
struct SVDResult;
struct EigenResult;

class Matrix {
public:
	Matrix() = default;
	Matrix(std::size_t rows, std::size_t cols);
	// device_only skips the host mirror until download()/host access is needed.
	Matrix(std::size_t rows, std::size_t cols, MemoryMode mode);
	Matrix(std::size_t rows, std::size_t cols, const std::vector<float>& values);
	Matrix(std::initializer_list<std::initializer_list<float>> values);

	Matrix(const Matrix& other);
	Matrix& operator=(const Matrix& other);
	Matrix(Matrix&& other) noexcept;
	Matrix& operator=(Matrix&& other) noexcept;
	~Matrix();

	std::size_t rows() const noexcept { return rows_; }
	std::size_t cols() const noexcept { return cols_; }
	std::size_t size() const noexcept { return rows_ * cols_; }
	bool empty() const noexcept { return size() == 0; }
	bool device_only() const noexcept { return device_only_; }
	bool host_materialized() const noexcept { return host_data_.size() == rows_ * cols_; }
	// Fail-fast stale-mirror contract: host_current() is false whenever the
	// device buffer holds newer data than host_data_. Every GPU-writing
	// operation calls mark_host_stale() on its output, and the host accessors
	// (at()/data()) throw instead of silently returning stale values; call
	// download() to refresh the mirror first.
	bool host_current() const noexcept { return host_current_; }
	void mark_host_stale() noexcept { host_current_ = false; }
	void mark_host_current() noexcept { host_current_ = true; }

	float& at(std::size_t row, std::size_t col);
	const float& at(std::size_t row, std::size_t col) const;
	const std::vector<float>& data() const;
	std::vector<float>& data() noexcept { return host_data_; }

	void fill(float value);
	void upload();
	void download();
	void synchronize() const;
	float* device_data() noexcept { return device_data_.get(); }
	const float* device_data() const noexcept { return device_data_.get(); }

	// --- transforms ---
	Matrix transpose() const;
	Matrix relu() const;
	Matrix softmax() const;
	Matrix flatten() const;
	Matrix reshape(std::size_t rows, std::size_t cols) const;
	Matrix slice(std::size_t row_start, std::size_t row_end,
		std::size_t col_start, std::size_t col_end) const;

	// --- elementwise math ---
	Matrix exp() const;
	Matrix log() const;
	Matrix sqrt() const;
	Matrix abs() const;
	Matrix clamp(float low, float high) const;
	Matrix sigmoid() const;
	Matrix tanh() const;
	Matrix leaky_relu(float negative_slope = 0.01f) const;
	Matrix elu(float alpha = 1.0f) const;
	Matrix gelu() const;
	Matrix swish(float beta = 1.0f) const;
	Matrix pow(float exponent) const;
	Matrix add_scalar(float value) const;

	// --- operators ---
	Matrix operator+() const;
	Matrix operator-() const;
	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Matrix operator*(const Matrix& other) const;
	Matrix operator*(float scalar) const;
	Matrix operator/(const Matrix& other) const;
	Matrix operator/(float scalar) const;
	Matrix elementwise_multiply(const Matrix& other) const;
	Matrix& operator+=(const Matrix& other);
	Matrix& operator-=(const Matrix& other);
	Matrix& operator*=(float scalar);
	Matrix& operator/=(float scalar);

	Matrix outer_product(const Matrix& other) const;
	Matrix kron(const Matrix& other) const;

	// --- broadcast style (numPy-style: v matches rows OR cols) ---
	Matrix add_row_vector(const Matrix& v) const;
	Matrix add_col_vector(const Matrix& v) const;
	Matrix multiply_row_vector(const Matrix& v) const;
	Matrix multiply_col_vector(const Matrix& v) const;
	Matrix row_sum() const;
	Matrix col_sum() const;
	// --- reductions ---
	float sum() const;
	float mean() const;
	float min() const;
	float max() const;
	std::size_t argmin() const;
	std::size_t argmax() const;
	float variance() const;
	float stddev() const;
	float l1_norm() const;
	float l2_norm() const;
	float frobenius_norm() const;
	float abs_max() const;
	float trace() const;
	float determinant() const;
	Matrix inverse() const;
	float condition_number() const;
	Matrix covariance() const;
	Matrix correlation() const;

	// --- logical & comparison operations ---
	Matrix greater(float value) const;
	Matrix greater(const Matrix& other) const;
	Matrix less(float value) const;
	Matrix less(const Matrix& other) const;
	Matrix equal(float value) const;
	Matrix equal(const Matrix& other) const;
	Matrix not_equal(float value) const;
	Matrix not_equal(const Matrix& other) const;
	Matrix logical_and(const Matrix& other) const;
	Matrix logical_or(const Matrix& other) const;
	Matrix logical_not() const;
	Matrix isnan() const;
	Matrix isinf() const;
	Matrix is_finite() const;
	bool any() const;
	bool all() const;
	Matrix apply_mask(const Matrix& mask, float value) const;
	Matrix filter_by_mask(const Matrix& mask) const;
	static Matrix where(const Matrix& condition, const Matrix& true_value, const Matrix& false_value);
	static Matrix where(const Matrix& condition, float true_value, float false_value);

	// --- advanced linear algebra (cuSOLVER-accelerated) ---
	Matrix solve(const Matrix& rhs) const;
	QRResult qr() const;
	SVDResult svd() const;
	Matrix cholesky() const;
	EigenResult eigen() const;
	Matrix pinv() const;
	std::size_t rank() const;
	Matrix solve_least_squares(const Matrix& rhs) const;

	// --- factories ---
	static Matrix zeros(std::size_t rows, std::size_t cols);
	static Matrix ones(std::size_t rows, std::size_t cols);
	static Matrix identity(std::size_t size);
	static Matrix random(std::size_t rows, std::size_t cols);   // uniform [0,1)
	static Matrix random(std::size_t rows, std::size_t cols, std::uint32_t seed);
	static Matrix uniform(std::size_t rows, std::size_t cols, float low, float high);
	static Matrix uniform(std::size_t rows, std::size_t cols, float low, float high, std::uint32_t seed);
	static Matrix randn(std::size_t rows, std::size_t cols);   // standard normal
	static Matrix randn(std::size_t rows, std::size_t cols, std::uint32_t seed);
	static Matrix glorot(std::size_t rows, std::size_t cols);  // Xavier init

	// --- persistence ---
	void save(const std::string& filename) const;
	static Matrix load(const std::string& filename);

private:
	std::size_t rows_ = 0;
	std::size_t cols_ = 0;
	std::vector<float> host_data_;
	std::unique_ptr<float, void (*)(float*)> device_data_{nullptr, nullptr};
	bool host_pinned_ = false;
	bool device_only_ = false;
	// Fail-fast stale-mirror flag (see host_current() above). True only while
	// host_data_ reflects the device buffer; GPU-writing operations flip it.
	bool host_current_ = true;

	void allocate_device();
	void ensure_host_buffer();
	void ensure_host_pinned();
	void release_host_pinned();
	void require_host_current(const char* accessor) const;
	void validate_same_shape(const Matrix& other) const;
	void validate_index(std::size_t row, std::size_t col) const;
};

Matrix operator*(float scalar, const Matrix& matrix);

}

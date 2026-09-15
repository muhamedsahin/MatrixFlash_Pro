#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

namespace matrix_pro {

class Matrix {
public:
	Matrix() = default;
	Matrix(std::size_t rows, std::size_t cols);
	Matrix(std::size_t rows, std::size_t cols, const std::vector<float>& values);
	Matrix(std::initializer_list<std::initializer_list<float>> values);

	Matrix(const Matrix& other);
	Matrix& operator=(const Matrix& other);
	Matrix(Matrix&& other) noexcept = default;
	Matrix& operator=(Matrix&& other) noexcept = default;
	~Matrix() = default;

	std::size_t rows() const noexcept { return rows_; }
	std::size_t cols() const noexcept { return cols_; }
	std::size_t size() const noexcept { return rows_ * cols_; }
	bool empty() const noexcept { return size() == 0; }

	float& at(std::size_t row, std::size_t col);
	const float& at(std::size_t row, std::size_t col) const;
	const std::vector<float>& data() const noexcept { return host_data_; }
	std::vector<float>& data() noexcept { return host_data_; }

	void fill(float value);
	void upload();
	void download();
	float* device_data() noexcept { return device_data_.get(); }
	const float* device_data() const noexcept { return device_data_.get(); }

	Matrix transpose() const;
	Matrix relu() const;
	Matrix softmax() const;
	Matrix flatten() const;
	Matrix slice(std::size_t row_start, std::size_t row_end,
		std::size_t col_start, std::size_t col_end) const;
	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Matrix operator*(const Matrix& other) const;
	Matrix operator*(float scalar) const;
	Matrix elementwise_multiply(const Matrix& other) const;
	float sum() const;
	float mean() const;
	float l2_norm() const;
	float trace() const;
	float determinant() const;
	Matrix inverse() const;

	static Matrix zeros(std::size_t rows, std::size_t cols);
	static Matrix ones(std::size_t rows, std::size_t cols);
	static Matrix identity(std::size_t size);

private:
	std::size_t rows_ = 0;
	std::size_t cols_ = 0;
	std::vector<float> host_data_;
	std::unique_ptr<float, void (*)(float*)> device_data_{nullptr, nullptr};

	void allocate_device();
	void validate_same_shape(const Matrix& other) const;
	void validate_index(std::size_t row, std::size_t col) const;
};

Matrix operator*(float scalar, const Matrix& matrix);

}

#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace matrix_pro {
namespace {

enum class CompareOp { greater, less, equal, not_equal };
enum class LogicalOp { and_op, or_op, not_op };

__global__ void compare_kernel(const float* input, float* output, std::size_t count, float scalar, CompareOp op) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = input[index];
    switch (op) {
        case CompareOp::greater: output[index] = value > scalar ? 1.0f : 0.0f; break;
        case CompareOp::less: output[index] = value < scalar ? 1.0f : 0.0f; break;
        case CompareOp::equal: output[index] = value == scalar ? 1.0f : 0.0f; break;
        case CompareOp::not_equal: output[index] = value != scalar ? 1.0f : 0.0f; break;
    }
}

__global__ void compare_pair_kernel(const float* left, const float* right, float* output,
                                   std::size_t count, CompareOp op) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float a = left[index];
    const float b = right[index];
    switch (op) {
        case CompareOp::greater: output[index] = a > b ? 1.0f : 0.0f; break;
        case CompareOp::less: output[index] = a < b ? 1.0f : 0.0f; break;
        case CompareOp::equal: output[index] = a == b ? 1.0f : 0.0f; break;
        case CompareOp::not_equal: output[index] = a != b ? 1.0f : 0.0f; break;
    }
}

__global__ void logical_kernel(const float* left, const float* right, float* output,
                              std::size_t count, LogicalOp op) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const bool a = left[index] != 0.0f;
    const bool b = right[index] != 0.0f;
    switch (op) {
        case LogicalOp::and_op: output[index] = (a && b) ? 1.0f : 0.0f; break;
        case LogicalOp::or_op: output[index] = (a || b) ? 1.0f : 0.0f; break;
        case LogicalOp::not_op: output[index] = (!a) ? 1.0f : 0.0f; break;
    }
}

__global__ void unary_predicate_kernel(const float* input, float* output, std::size_t count, int mode) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = input[index];
    if (mode == 0) output[index] = ::isnan(value) ? 1.0f : 0.0f;
    else if (mode == 1) output[index] = ::isinf(value) ? 1.0f : 0.0f;
    else output[index] = ::isfinite(value) ? 1.0f : 0.0f;
}

__global__ void where_kernel(const float* condition, const float* true_value, const float* false_value,
                            float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = condition[index] != 0.0f ? true_value[index] : false_value[index];
}

__global__ void where_scalar_kernel(const float* condition, float true_value, float false_value,
                                   float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = condition[index] != 0.0f ? true_value : false_value;
}

__global__ void apply_mask_kernel(const float* input, const float* mask, float* output,
                                 std::size_t count, float value) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = mask[index] != 0.0f ? value : input[index];
}

__global__ void any_kernel(const float* input, std::size_t count, unsigned int* result) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && input[index] != 0.0f) atomicOr(result, 1U);
}

__global__ void all_kernel(const float* input, std::size_t count, unsigned int* result) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && input[index] == 0.0f) atomicAnd(result, 0U);
}

__global__ void count_mask_kernel(const float* mask, std::size_t count, unsigned long long* result) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && mask[index] != 0.0f) atomicAdd(result, 1ULL);
}

__global__ void scatter_mask_kernel(const float* input, const float* mask, float* output,
                                    std::size_t count, unsigned long long* position) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && mask[index] != 0.0f) {
        const auto destination = atomicAdd(position, 1ULL);
        output[destination] = input[index];
    }
}

}

Matrix greater(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), scalar, CompareOp::greater);
    checkCuda(cudaGetLastError(), "greater kernel launch");
    return output;
}

Matrix greater(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("greater() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), CompareOp::greater);
    checkCuda(cudaGetLastError(), "greater pair kernel launch");
    return output;
}

Matrix less(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), scalar, CompareOp::less);
    checkCuda(cudaGetLastError(), "less kernel launch");
    return output;
}

Matrix less(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("less() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), CompareOp::less);
    checkCuda(cudaGetLastError(), "less pair kernel launch");
    return output;
}

Matrix equal(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), scalar, CompareOp::equal);
    checkCuda(cudaGetLastError(), "equal kernel launch");
    return output;
}

Matrix equal(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("equal() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), CompareOp::equal);
    checkCuda(cudaGetLastError(), "equal pair kernel launch");
    return output;
}

Matrix not_equal(const Matrix& matrix, float scalar) {
    Matrix output(matrix.rows(), matrix.cols());
    compare_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), scalar, CompareOp::not_equal);
    checkCuda(cudaGetLastError(), "not_equal kernel launch");
    return output;
}

Matrix not_equal(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("not_equal() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    compare_pair_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), CompareOp::not_equal);
    checkCuda(cudaGetLastError(), "not_equal pair kernel launch");
    return output;
}

Matrix logical_and(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("logical_and() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    logical_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), LogicalOp::and_op);
    checkCuda(cudaGetLastError(), "logical_and kernel launch");
    return output;
}

Matrix logical_or(const Matrix& left, const Matrix& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        throw std::invalid_argument("logical_or() requires matching shapes");
    }
    Matrix output(left.rows(), left.cols());
    logical_kernel<<<static_cast<unsigned>((left.size() + 255) / 256), 256, 0, compute_stream()>>>(
        left.device_data(), right.device_data(), output.device_data(), left.size(), LogicalOp::or_op);
    checkCuda(cudaGetLastError(), "logical_or kernel launch");
    return output;
}

Matrix logical_not(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    logical_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), matrix.device_data(), output.device_data(), matrix.size(), LogicalOp::not_op);
    checkCuda(cudaGetLastError(), "logical_not kernel launch");
    return output;
}

Matrix isnan(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    unary_predicate_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), 0);
    checkCuda(cudaGetLastError(), "isnan kernel launch");
    return output;
}

Matrix isinf(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    unary_predicate_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), 1);
    checkCuda(cudaGetLastError(), "isinf kernel launch");
    return output;
}

Matrix is_finite(const Matrix& matrix) {
    Matrix output(matrix.rows(), matrix.cols());
    unary_predicate_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), output.device_data(), matrix.size(), 2);
    checkCuda(cudaGetLastError(), "is_finite kernel launch");
    return output;
}

bool any(const Matrix& matrix) {
    if (matrix.empty()) return false;
    auto* result = static_cast<unsigned int*>(allocate_device_memory(sizeof(unsigned int)));
    checkCuda(cudaMemsetAsync(result, 0, sizeof(unsigned int), compute_stream()), "any result reset");
    any_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(matrix.device_data(), matrix.size(), result);
    checkCuda(cudaGetLastError(), "any kernel launch");
    unsigned int host_result = 0;
    checkCuda(cudaMemcpyAsync(&host_result, result, sizeof(host_result), cudaMemcpyDeviceToHost, compute_stream()), "any result read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "any synchronize");
    free_device_memory(result);
    return host_result != 0;
}

bool all(const Matrix& matrix) {
    if (matrix.empty()) return true;
    auto* result = static_cast<unsigned int*>(allocate_device_memory(sizeof(unsigned int)));
    checkCuda(cudaMemsetAsync(result, 1, sizeof(unsigned int), compute_stream()), "all result reset");
    all_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(matrix.device_data(), matrix.size(), result);
    checkCuda(cudaGetLastError(), "all kernel launch");
    unsigned int host_result = 0;
    checkCuda(cudaMemcpyAsync(&host_result, result, sizeof(host_result), cudaMemcpyDeviceToHost, compute_stream()), "all result read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "all synchronize");
    free_device_memory(result);
    return host_result != 0;
}

Matrix where(const Matrix& condition, const Matrix& true_value, const Matrix& false_value) {
    if (condition.rows() != true_value.rows() || condition.cols() != true_value.cols() ||
        condition.rows() != false_value.rows() || condition.cols() != false_value.cols()) {
        throw std::invalid_argument("where() requires matching condition/branch shapes");
    }
    Matrix output(condition.rows(), condition.cols());
    where_kernel<<<static_cast<unsigned>((condition.size() + 255) / 256), 256, 0, compute_stream()>>>(
        condition.device_data(), true_value.device_data(), false_value.device_data(), output.device_data(), condition.size());
    checkCuda(cudaGetLastError(), "where kernel launch");
    return output;
}

Matrix where(const Matrix& condition, float true_value, float false_value) {
    Matrix output(condition.rows(), condition.cols());
    where_scalar_kernel<<<static_cast<unsigned>((condition.size() + 255) / 256), 256, 0, compute_stream()>>>(
        condition.device_data(), true_value, false_value, output.device_data(), condition.size());
    checkCuda(cudaGetLastError(), "where scalar kernel launch");
    return output;
}

Matrix apply_mask(const Matrix& matrix, const Matrix& mask, float value) {
    if (matrix.rows() != mask.rows() || matrix.cols() != mask.cols()) {
        throw std::invalid_argument("apply_mask() requires matching shapes");
    }
    Matrix output(matrix.rows(), matrix.cols());
    apply_mask_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), mask.device_data(), output.device_data(), matrix.size(), value);
    checkCuda(cudaGetLastError(), "apply_mask kernel launch");
    return output;
}

Matrix filter_by_mask(const Matrix& matrix, const Matrix& mask) {
    if (matrix.rows() != mask.rows() || matrix.cols() != mask.cols()) {
        throw std::invalid_argument("filter_by_mask() requires matching shapes");
    }
    const std::size_t count = matrix.size();
    auto* device_count = static_cast<unsigned long long*>(allocate_device_memory(sizeof(unsigned long long)));
    checkCuda(cudaMemsetAsync(device_count, 0, sizeof(unsigned long long), compute_stream()), "filter count reset");
    count_mask_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, compute_stream()>>>(mask.device_data(), count, device_count);
    checkCuda(cudaGetLastError(), "filter count kernel launch");
    unsigned long long active_count = 0;
    checkCuda(cudaMemcpyAsync(&active_count, device_count, sizeof(active_count), cudaMemcpyDeviceToHost, compute_stream()), "filter count read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "filter count synchronize");
    free_device_memory(device_count);
    const std::size_t active = static_cast<std::size_t>(active_count);
    Matrix output(active, 1);
    if (active == 0) return output;

    auto* device_position = static_cast<unsigned long long*>(allocate_device_memory(sizeof(unsigned long long)));
    checkCuda(cudaMemsetAsync(device_position, 0, sizeof(unsigned long long), compute_stream()), "filter position reset");
    scatter_mask_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), mask.device_data(), output.device_data(), count, device_position);
    checkCuda(cudaGetLastError(), "filter scatter kernel launch");
    free_device_memory(device_position);
    return output;
}

Matrix Matrix::greater(float value) const { return matrix_pro::greater(*this, value); }
Matrix Matrix::greater(const Matrix& other) const { return matrix_pro::greater(*this, other); }
Matrix Matrix::less(float value) const { return matrix_pro::less(*this, value); }
Matrix Matrix::less(const Matrix& other) const { return matrix_pro::less(*this, other); }
Matrix Matrix::equal(float value) const { return matrix_pro::equal(*this, value); }
Matrix Matrix::equal(const Matrix& other) const { return matrix_pro::equal(*this, other); }
Matrix Matrix::not_equal(float value) const { return matrix_pro::not_equal(*this, value); }
Matrix Matrix::not_equal(const Matrix& other) const { return matrix_pro::not_equal(*this, other); }
Matrix Matrix::logical_and(const Matrix& other) const { return matrix_pro::logical_and(*this, other); }
Matrix Matrix::logical_or(const Matrix& other) const { return matrix_pro::logical_or(*this, other); }
Matrix Matrix::logical_not() const { return matrix_pro::logical_not(*this); }
Matrix Matrix::isnan() const { return matrix_pro::isnan(*this); }
Matrix Matrix::isinf() const { return matrix_pro::isinf(*this); }
Matrix Matrix::is_finite() const { return matrix_pro::is_finite(*this); }
bool Matrix::any() const { return matrix_pro::any(*this); }
bool Matrix::all() const { return matrix_pro::all(*this); }
Matrix Matrix::apply_mask(const Matrix& mask, float value) const { return matrix_pro::apply_mask(*this, mask, value); }
Matrix Matrix::filter_by_mask(const Matrix& mask) const { return matrix_pro::filter_by_mask(*this, mask); }
Matrix Matrix::where(const Matrix& condition, const Matrix& true_value, const Matrix& false_value) { return matrix_pro::where(condition, true_value, false_value); }
Matrix Matrix::where(const Matrix& condition, float true_value, float false_value) { return matrix_pro::where(condition, true_value, false_value); }

} // namespace matrix_pro

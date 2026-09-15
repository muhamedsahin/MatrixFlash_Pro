#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

namespace matrix_pro {
namespace {

constexpr unsigned reduce_block = 256;

__global__ void reduce_partial_kernel(const float* input, std::size_t count, float* partial) {
    __shared__ float ssum[reduce_block];
    __shared__ float ssq[reduce_block];
    __shared__ float sl1[reduce_block];
    __shared__ float smin[reduce_block];
    __shared__ float smax[reduce_block];
    const auto tid = threadIdx.x;

    float sum = 0.0f, sum_sq = 0.0f, sum_l1 = 0.0f;
    float local_min = FLT_MAX, local_max = -FLT_MAX;
    for (std::size_t index = blockIdx.x * reduce_block + tid; index < count; index += gridDim.x * reduce_block) {
        const float value = input[index];
        sum += value;
        sum_l1 += fabsf(value);
        sum_sq += value * value;
        local_min = fminf(local_min, value);
        local_max = fmaxf(local_max, value);
    }

    ssum[tid] = sum; ssq[tid] = sum_sq; sl1[tid] = sum_l1;
    smin[tid] = local_min; smax[tid] = local_max;
    __syncthreads();

    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
            ssq[tid] += ssq[tid + stride];
            sl1[tid] += sl1[tid + stride];
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * 5U;
        partial[offset + 0] = ssum[0];
        partial[offset + 1] = ssq[0];
        partial[offset + 2] = sl1[0];
        partial[offset + 3] = smin[0];
        partial[offset + 4] = smax[0];
    }
}

__global__ void arg_partial_kernel(const float* input, std::size_t count, int find_max,
                                  float* partial_values, std::size_t* partial_indices) {
    __shared__ float svalue[reduce_block];
    __shared__ std::size_t sindex[reduce_block];

    const auto tid = threadIdx.x;
    float best = find_max ? -FLT_MAX : FLT_MAX;
    std::size_t best_index = 0;

    for (std::size_t index = blockIdx.x * reduce_block + tid; index < count; index += gridDim.x * reduce_block) {
        const float value = input[index];
        if (find_max ? (value > best) : (value < best)) {
            best = value;
            best_index = index;
        }
    }

    svalue[tid] = best;
    sindex[tid] = best_index;
    __syncthreads();

    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float va = svalue[tid];
            const float vb = svalue[tid + stride];
            const std::size_t ia = sindex[tid];
            const std::size_t ib = sindex[tid + stride];
            const bool better = find_max ? (vb > va) : (vb < va);
            const bool tie = (vb == va);
            if (better || (tie && ib < ia)) {
                svalue[tid] = vb;
                sindex[tid] = ib;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_values[blockIdx.x] = svalue[0];
        partial_indices[blockIdx.x] = sindex[0];
    }
}

__global__ void row_sum_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
    for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col) sum += input[row * cols + col];
        output[row] = sum;
    }
}

__global__ void col_sum_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
    for (std::size_t col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (std::size_t row = 0; row < rows; ++row) sum += input[row * cols + col];
        output[col] = sum;
    }
}

}

float reduce_scalar(const Matrix& matrix, int which) {
    const std::size_t count = matrix.size();
    if (count == 0) {
        if (which == 3) return FLT_MAX;
        if (which == 4) return -FLT_MAX;
        return 0.0f;
    }

    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);
    float* partial = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * 5U * sizeof(float)));
    reduce_partial_kernel<<<grid, reduce_block>>>(matrix.device_data(), count, partial);
    checkCuda(cudaGetLastError(), "reduce partial kernel launch");

    std::vector<float> host_partial(static_cast<std::size_t>(grid) * 5U);
    checkCuda(cudaMemcpy(host_partial.data(), partial, host_partial.size() * sizeof(float), cudaMemcpyDeviceToHost), "reduce partial read");
    free_device_memory(partial);

    float result[5] = {0.0f, 0.0f, 0.0f, FLT_MAX, -FLT_MAX};
    for (std::size_t block = 0; block < static_cast<std::size_t>(grid); ++block) {
        const std::size_t offset = block * 5U;
        result[0] += host_partial[offset + 0];
        result[1] += host_partial[offset + 1];
        result[2] += host_partial[offset + 2];
        result[3] = std::fmin(result[3], host_partial[offset + 3]);
        result[4] = std::fmax(result[4], host_partial[offset + 4]);
    }
    return result[which];
}

float sum(const Matrix& matrix) { return reduce_scalar(matrix, 0); }
float mean(const Matrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("Mean is undefined for an empty matrix");
    return sum(matrix) / static_cast<float>(matrix.size());
}
float min(const Matrix& matrix) { return reduce_scalar(matrix, 3); }
float max(const Matrix& matrix) { return reduce_scalar(matrix, 4); }
float l1_norm(const Matrix& matrix) { return reduce_scalar(matrix, 2); }
float abs_max(const Matrix& matrix) { return fabsf(reduce_scalar(matrix, 4)); }
float l2_norm(const Matrix& matrix) { return std::sqrt(reduce_scalar(matrix, 1)); }

float variance(const Matrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("Variance is undefined for an empty matrix");
    const std::size_t count = matrix.size();
    if (count == 1) return 0.0f;

    const float total = sum(matrix);
    const float sum_sq = reduce_scalar(matrix, 1);
    const float mean_value = total / static_cast<float>(count);
    const float mean_sq = mean_value * mean_value;
    const float numerator = static_cast<float>(count) * sum_sq - total * total;
    return numerator / (static_cast<float>(count) * static_cast<float>(count - 1));
}
float stddev(const Matrix& matrix) { return std::sqrt(variance(matrix)); }

std::size_t argmin(const Matrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("argmin undefined for empty matrix");
    const std::size_t count = matrix.size();
    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);

    float* partial_values = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    std::size_t* partial_indices = static_cast<std::size_t*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(std::size_t)));
    arg_partial_kernel<<<grid, reduce_block>>>(matrix.device_data(), count, 0, partial_values, partial_indices);
    checkCuda(cudaGetLastError(), "argmin partial kernel launch");

    std::vector<float> host_values(static_cast<std::size_t>(grid));
    std::vector<std::size_t> host_indices(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpy(host_values.data(), partial_values, host_values.size() * sizeof(float), cudaMemcpyDeviceToHost), "argmin values read");
    checkCuda(cudaMemcpy(host_indices.data(), partial_indices, host_indices.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost), "argmin indices read");
    free_device_memory(partial_values);
    free_device_memory(partial_indices);

    std::size_t best_index = host_indices.front();
    float best_value = host_values.front();
    for (std::size_t i = 1; i < host_values.size(); ++i) {
        if (host_values[i] < best_value || (host_values[i] == best_value && host_indices[i] < best_index)) {
            best_index = host_indices[i];
            best_value = host_values[i];
        }
    }
    return best_index;
}

std::size_t argmax(const Matrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("argmax undefined for empty matrix");
    const std::size_t count = matrix.size();
    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);

    float* partial_values = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    std::size_t* partial_indices = static_cast<std::size_t*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(std::size_t)));
    arg_partial_kernel<<<grid, reduce_block>>>(matrix.device_data(), count, 1, partial_values, partial_indices);
    checkCuda(cudaGetLastError(), "argmax partial kernel launch");

    std::vector<float> host_values(static_cast<std::size_t>(grid));
    std::vector<std::size_t> host_indices(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpy(host_values.data(), partial_values, host_values.size() * sizeof(float), cudaMemcpyDeviceToHost), "argmax values read");
    checkCuda(cudaMemcpy(host_indices.data(), partial_indices, host_indices.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost), "argmax indices read");
    free_device_memory(partial_values);
    free_device_memory(partial_indices);

    std::size_t best_index = host_indices.front();
    float best_value = host_values.front();
    for (std::size_t i = 1; i < host_values.size(); ++i) {
        if (host_values[i] > best_value || (host_values[i] == best_value && host_indices[i] < best_index)) {
            best_index = host_indices[i];
            best_value = host_values[i];
        }
    }
    return best_index;
}

__global__ void trace_kernel(const float* input, std::size_t rows, std::size_t cols, float* output) {
    __shared__ float shared[reduce_block];
    const auto tid = threadIdx.x;
    float local = 0.0f;
    for (std::size_t index = tid; index < rows; index += blockDim.x) {
        const std::size_t row_index = index * cols + index;
        local += input[row_index];
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *output = shared[0];
}

float trace(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) throw std::invalid_argument("Trace requires a square matrix");
    if (matrix.empty()) return 0.0f;
    float* device_result = static_cast<float*>(allocate_device_memory(sizeof(float)));
    trace_kernel<<<1, reduce_block>>>(matrix.device_data(), matrix.rows(), matrix.cols(), device_result);
    checkCuda(cudaGetLastError(), "trace kernel launch");
    float result = 0.0f;
    checkCuda(cudaMemcpy(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost), "trace read");
    free_device_memory(device_result);
    return result;
}

Matrix row_sum(const Matrix& matrix) {
    Matrix output(matrix.rows(), 1);
    const unsigned grid = matrix.rows() == 0 ? 1 : static_cast<unsigned>((matrix.rows() + 255) / 256);
    row_sum_kernel<<<grid, 256>>>(matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "row sum kernel launch");
    return output;
}

Matrix col_sum(const Matrix& matrix) {
    Matrix output(1, matrix.cols());
    const unsigned grid = matrix.cols() == 0 ? 1 : static_cast<unsigned>((matrix.cols() + 255) / 256);
    col_sum_kernel<<<grid, 256>>>(matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "col sum kernel launch");
    return output;
}

float Matrix::sum() const { return matrix_pro::sum(*this); }
float Matrix::mean() const { return matrix_pro::mean(*this); }
float Matrix::min() const { return matrix_pro::min(*this); }
float Matrix::max() const { return matrix_pro::max(*this); }
std::size_t Matrix::argmin() const { return matrix_pro::argmin(*this); }
std::size_t Matrix::argmax() const { return matrix_pro::argmax(*this); }
float Matrix::variance() const { return matrix_pro::variance(*this); }
float Matrix::stddev() const { return matrix_pro::stddev(*this); }
float Matrix::l1_norm() const { return matrix_pro::l1_norm(*this); }
float Matrix::l2_norm() const { return matrix_pro::l2_norm(*this); }
float Matrix::abs_max() const { return matrix_pro::abs_max(*this); }
float Matrix::trace() const { return matrix_pro::trace(*this); }
Matrix Matrix::row_sum() const { return matrix_pro::row_sum(*this); }
Matrix Matrix::col_sum() const { return matrix_pro::col_sum(*this); }

}
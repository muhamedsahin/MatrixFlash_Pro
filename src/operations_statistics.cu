#include "matrix_pro/operations.hpp"
#include "matrix_pro/cuda_utils.hpp"

#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

namespace matrix_pro {
namespace {

constexpr unsigned reduce_block = 256;

// Aggregate buffer: [0]=sum, [1]=sumSq, [2]=l1, [3]=min, [4]=max
__device__ float g_agg[5];
__device__ float g_best_value;
__device__ std::size_t g_best_index;

__device__ float atomicMaxf(float* address, float value) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float atomicMinf(float* address, float value) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void reduce_kernel(const float* input, std::size_t count) {
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
        atomicAdd(&g_agg[0], ssum[0]);
        atomicAdd(&g_agg[1], ssq[0]);
        atomicAdd(&g_agg[2], sl1[0]);
        atomicMinf(&g_agg[3], smin[0]);
        atomicMaxf(&g_agg[4], smax[0]);
    }
}

__global__ void arg_reduce_kernel(const float* input, std::size_t count, int find_max) {
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
        g_best_index = sindex[0];
        g_best_value = svalue[0];
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
    const float init[] = {0.0f, 0.0f, 0.0f, FLT_MAX, -FLT_MAX};
    checkCuda(cudaMemcpyToSymbol(g_agg, init, sizeof(init), 0, cudaMemcpyHostToDevice), "reduce reset");
    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);
    reduce_kernel<<<grid, reduce_block>>>(matrix.device_data(), count);
    checkCuda(cudaGetLastError(), "reduce kernel launch");
    float host[5];
    checkCuda(cudaMemcpyFromSymbol(host, g_agg, sizeof(host), 0, cudaMemcpyDeviceToHost), "reduce read");
    return host[which];
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
    std::size_t result = 0;
    checkCuda(cudaMemcpyToSymbol(g_best_index, &result, sizeof(result), 0, cudaMemcpyHostToDevice), "arg reset");
    const std::size_t count = matrix.size();
    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);
    arg_reduce_kernel<<<grid, reduce_block>>>(matrix.device_data(), count, 0);
    checkCuda(cudaGetLastError(), "argmin kernel launch");
    checkCuda(cudaMemcpyFromSymbol(&result, g_best_index, sizeof(result), 0, cudaMemcpyDeviceToHost), "argmin read");
    return result;
}

std::size_t argmax(const Matrix& matrix) {
    if (matrix.empty()) throw std::invalid_argument("argmax undefined for empty matrix");
    std::size_t result = 0;
    checkCuda(cudaMemcpyToSymbol(g_best_index, &result, sizeof(result), 0, cudaMemcpyHostToDevice), "arg reset");
    const std::size_t count = matrix.size();
    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);
    arg_reduce_kernel<<<grid, reduce_block>>>(matrix.device_data(), count, 1);
    checkCuda(cudaGetLastError(), "argmax kernel launch");
    checkCuda(cudaMemcpyFromSymbol(&result, g_best_index, sizeof(result), 0, cudaMemcpyDeviceToHost), "argmax read");
    return result;
}

float trace(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) throw std::invalid_argument("Trace requires a square matrix");
    if (matrix.empty()) return 0.0f;
    Matrix copy = matrix; copy.download();
    float result = 0.0f;
    for (std::size_t index = 0; index < matrix.rows(); ++index) result += copy.at(index, index);
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
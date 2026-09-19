#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cublas_v2.h>

#include <array>
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

namespace matrix_pro {
namespace {

constexpr unsigned reduce_block = 256;

// Pinned host scratch for device->host partial reads: pageable copies serialize
// the stream, pinned copies go straight over DMA. Falls back to normal memory
// when cudaMallocHost is unavailable (host without WDDM/pinning support).
class PinnedFloatScratch {
public:
    explicit PinnedFloatScratch(std::size_t count) : fallback_(count) {
        if (cudaMallocHost(&pinned_, count * sizeof(float)) != cudaSuccess) {
            pinned_ = nullptr;
            cudaGetLastError(); // clear the sticky error; fallback covers us
        }
    }
    ~PinnedFloatScratch() {
        if (pinned_ != nullptr) cudaFreeHost(pinned_);
    }
    PinnedFloatScratch(const PinnedFloatScratch&) = delete;
    PinnedFloatScratch& operator=(const PinnedFloatScratch&) = delete;
    float* data() noexcept { return pinned_ != nullptr ? pinned_ : fallback_.data(); }
private:
    float* pinned_ = nullptr;
    std::vector<float> fallback_;
};

// Reduction grids are capped: the kernels are grid-stride loops, so a small
// multiple of the SM count saturates the GPU while keeping the partial buffer
// (and the host-side combine pass) tiny instead of scaling with element count.
unsigned reduce_grid(std::size_t count) {
    const unsigned needed = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);
    const unsigned max_blocks = static_cast<unsigned>(matrix_pro::sm_count()) * 8u;
    return needed < max_blocks ? needed : max_blocks;
}

__global__ void reduce_partial_kernel(const float* input, std::size_t count, float* partial) {
    __shared__ float ssum[reduce_block];
    __shared__ float ssq[reduce_block];
    __shared__ float sl1[reduce_block];
    __shared__ float smin[reduce_block];
    __shared__ float smax[reduce_block];
    const auto tid = threadIdx.x;

    float sum = 0.0f, sum_sq = 0.0f, sum_l1 = 0.0f;
    float local_min = FLT_MAX, local_max = -FLT_MAX;

    // float4 prefix: 4x fewer global loads on aligned buffers.
    const std::size_t n4 = count / 4;
    if (n4 > 0 && (reinterpret_cast<uintptr_t>(input) & 0xFu) == 0u) {
        const float4* in4 = reinterpret_cast<const float4*>(input);
        for (std::size_t index = blockIdx.x * reduce_block + tid; index < n4;
             index += static_cast<std::size_t>(gridDim.x) * reduce_block) {
            const float4 v = in4[index];
            sum += v.x + v.y + v.z + v.w;
            sum_l1 += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            local_min = fminf(local_min, fminf(fminf(v.x, v.y), fminf(v.z, v.w)));
            local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
        for (std::size_t index = n4 * 4 + blockIdx.x * reduce_block + tid; index < count;
             index += static_cast<std::size_t>(gridDim.x) * reduce_block) {
            const float value = input[index];
            sum += value;
            sum_l1 += fabsf(value);
            sum_sq += value * value;
            local_min = fminf(local_min, value);
            local_max = fmaxf(local_max, value);
        }
    } else {
        for (std::size_t index = blockIdx.x * reduce_block + tid; index < count;
             index += static_cast<std::size_t>(gridDim.x) * reduce_block) {
            const float value = input[index];
            sum += value;
            sum_l1 += fabsf(value);
            sum_sq += value * value;
            local_min = fminf(local_min, value);
            local_max = fmaxf(local_max, value);
        }
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
    __shared__ float partial[reduce_block];
    const auto tid = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float sum = 0.0f;
        for (std::size_t col = tid; col < cols; col += blockDim.x) sum += input[row * cols + col];
        partial[tid] = sum;
        __syncthreads();
        for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0) output[row] = partial[0];
    }
}

__global__ void col_sum_kernel(const float* __restrict__ input, float* __restrict__ output, std::size_t rows, std::size_t cols) {
    // One thread per column: consecutive threads read consecutive addresses
    // within each row, so every row pass is a coalesced transaction. Four
    // independent accumulators break the serial dependency chain so loads
    // overlap instead of stalling on latency.
    for (std::size_t col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        std::size_t row = 0;
        for (; row + 4 <= rows; row += 4) {
            const float* base = input + row * cols + col;
            s0 += base[0];
            s1 += base[cols];
            s2 += base[2 * cols];
            s3 += base[3 * cols];
        }
        for (; row < rows; ++row) s0 += input[row * cols + col];
        output[col] = (s0 + s1) + (s2 + s3);
    }
}

__global__ void center_columns_kernel(const float* input, const float* means, float* output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows * cols) output[index] = input[index] - means[index % cols];
}

__global__ void sum_partial_double_kernel(const float* input, std::size_t count, double* partial) {
    __shared__ double shared[reduce_block];
    const auto tid = threadIdx.x;
    double local = 0.0;
    for (std::size_t index = blockIdx.x * reduce_block + tid; index < count; index += gridDim.x * reduce_block) {
        local += static_cast<double>(input[index]);
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shared[0];
}

__global__ void centered_sq_sum_kernel(const float* input, std::size_t count, float mean, float* output) {
    __shared__ float shared[reduce_block];
    const auto tid = threadIdx.x;
    float local = 0.0f;
    for (std::size_t index = blockIdx.x * reduce_block + tid; index < count; index += gridDim.x * reduce_block) {
        const float diff = input[index] - mean;
        local += diff * diff;
    }
    shared[tid] = local;
    __syncthreads();
    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared[0];
}
__global__ void covariance_to_correlation_kernel(const float* covariance, float* correlation,
                                                 std::size_t dimensions) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= dimensions * dimensions) return;
    const auto row = index / dimensions;
    const auto col = index % dimensions;
    if (row == col) {
        correlation[index] = 1.0f;
        return;
    }
    const float variance_row = covariance[row * dimensions + row];
    const float variance_col = covariance[col * dimensions + col];
    correlation[index] = variance_row > 0.0f && variance_col > 0.0f
        ? covariance[index] / sqrtf(variance_row * variance_col) : 0.0f;
}

}

std::array<float, 5> reduce_stats(const Matrix& matrix) {
    const std::size_t count = matrix.size();
    if (count == 0) {
        return {0.0f, 0.0f, 0.0f, FLT_MAX, -FLT_MAX};
    }

    const unsigned grid = reduce_grid(count);
    float* partial = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * 5U * sizeof(float)));
    reduce_partial_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), count, partial);
    checkCuda(cudaGetLastError(), "reduce partial kernel launch");

    PinnedFloatScratch host_partial(static_cast<std::size_t>(grid) * 5U);
    checkCuda(cudaMemcpyAsync(host_partial.data(), partial, static_cast<std::size_t>(grid) * 5U * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "reduce partial read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "reduce partial synchronize");
    free_device_memory(partial);

    const float* host = host_partial.data();

    std::array<float, 5> result = {0.0f, 0.0f, 0.0f, FLT_MAX, -FLT_MAX};
    for (std::size_t block = 0; block < static_cast<std::size_t>(grid); ++block) {
        const std::size_t offset = block * 5U;
        result[0] += host[offset + 0];
        result[1] += host[offset + 1];
        result[2] += host[offset + 2];
        result[3] = std::fmin(result[3], host[offset + 3]);
        result[4] = std::fmax(result[4], host[offset + 4]);
    }
    return result;
}

float reduce_scalar(const Matrix& matrix, int which) {
    return reduce_stats(matrix)[static_cast<std::size_t>(which)];
}

float sum(const Matrix& matrix) { return reduce_scalar(matrix, 0); }
float mean(const Matrix& matrix) {
    if (matrix.empty()) throw InvalidArgumentError("Mean is undefined for an empty matrix");
    return sum(matrix) / static_cast<float>(matrix.size());
}
float min(const Matrix& matrix) { return reduce_scalar(matrix, 3); }
float max(const Matrix& matrix) { return reduce_scalar(matrix, 4); }
float l1_norm(const Matrix& matrix) { return reduce_scalar(matrix, 2); }
float l2_norm(const Matrix& matrix) { return std::sqrt(reduce_scalar(matrix, 1)); }
float frobenius_norm(const Matrix& matrix) { return l2_norm(matrix); }
float abs_max(const Matrix& matrix) {
    if (matrix.empty()) return 0.0f;
    const std::array<float, 5> stats = reduce_stats(matrix);
    return std::fmax(std::fabs(stats[3]), std::fabs(stats[4]));
}

float variance(const Matrix& matrix) {
    if (matrix.empty()) throw InvalidArgumentError("Variance is undefined for an empty matrix");
    const std::size_t count = matrix.size();
    if (count == 1) return 0.0f;

    // Two-pass computation for numerical stability (avoids catastrophic cancellation).
    // The mean is accumulated in double precision: a naive float sum of large
    // inputs loses low-order bits, which would poison the second pass.
    const unsigned grid = reduce_grid(count);
    double* mean_partial = static_cast<double*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(double)));
    sum_partial_double_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), count, mean_partial);
    checkCuda(cudaGetLastError(), "double sum kernel launch");

    std::vector<double> host_mean_partial(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_mean_partial.data(), mean_partial, host_mean_partial.size() * sizeof(double), cudaMemcpyDeviceToHost, compute_stream()), "double sum read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "double sum synchronize");
    free_device_memory(mean_partial);

    double total = 0.0;
    for (double value : host_mean_partial) total += value;
    const float mean_value = static_cast<float>(total / static_cast<double>(count));

    float* partial = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    centered_sq_sum_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), count, mean_value, partial);
    checkCuda(cudaGetLastError(), "centered square sum kernel launch");

    std::vector<float> host_partial(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_partial.data(), partial, host_partial.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "centered square sum read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "centered square sum synchronize");
    free_device_memory(partial);

    float sum_squared_diff = 0.0f;
    for (float value : host_partial) sum_squared_diff += value;
    return sum_squared_diff / static_cast<float>(count - 1);
}
float stddev(const Matrix& matrix) { return std::sqrt(variance(matrix)); }

std::size_t argmin(const Matrix& matrix) {
    if (matrix.empty()) throw InvalidArgumentError("argmin undefined for empty matrix");
    const std::size_t count = matrix.size();
    const unsigned grid = reduce_grid(count);

    float* partial_values = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    std::size_t* partial_indices = static_cast<std::size_t*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(std::size_t)));
    arg_partial_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), count, 0, partial_values, partial_indices);
    checkCuda(cudaGetLastError(), "argmin partial kernel launch");

    std::vector<float> host_values(static_cast<std::size_t>(grid));
    std::vector<std::size_t> host_indices(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_values.data(), partial_values, host_values.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "argmin values read");
    checkCuda(cudaMemcpyAsync(host_indices.data(), partial_indices, host_indices.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost, compute_stream()), "argmin indices read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "argmin synchronize");
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
    if (matrix.empty()) throw InvalidArgumentError("argmax undefined for empty matrix");
    const std::size_t count = matrix.size();
    const unsigned grid = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);

    float* partial_values = static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(float)));
    std::size_t* partial_indices = static_cast<std::size_t*>(allocate_device_memory(static_cast<std::size_t>(grid) * sizeof(std::size_t)));
    arg_partial_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), count, 1, partial_values, partial_indices);
    checkCuda(cudaGetLastError(), "argmax partial kernel launch");

    std::vector<float> host_values(static_cast<std::size_t>(grid));
    std::vector<std::size_t> host_indices(static_cast<std::size_t>(grid));
    checkCuda(cudaMemcpyAsync(host_values.data(), partial_values, host_values.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "argmax values read");
    checkCuda(cudaMemcpyAsync(host_indices.data(), partial_indices, host_indices.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost, compute_stream()), "argmax indices read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "argmax synchronize");
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
    if (matrix.rows() != matrix.cols()) throw InvalidArgumentError("Trace requires a square matrix");
    if (matrix.empty()) return 0.0f;
    float* device_result = static_cast<float*>(allocate_device_memory(sizeof(float)));
    trace_kernel<<<1, reduce_block, 0, compute_stream()>>>(matrix.device_data(), matrix.rows(), matrix.cols(), device_result);
    checkCuda(cudaGetLastError(), "trace kernel launch");
    float result = 0.0f;
    checkCuda(cudaMemcpyAsync(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "trace read");
    checkCuda(cudaStreamSynchronize(compute_stream()), "trace synchronize");
    free_device_memory(device_result);
    return result;
}

Matrix row_sum(const Matrix& matrix) {
    Matrix output(matrix.rows(), 1);
    // Cap the grid (SM count * 4) and loop rows inside the kernel so very tall
    // matrices don't launch absurd block counts.
    unsigned grid = matrix.rows() == 0 ? 1 : static_cast<unsigned>(matrix.rows());
    const unsigned max_blocks = static_cast<unsigned>(matrix_pro::sm_count()) * 4u;
    if (grid > max_blocks) grid = max_blocks;
    row_sum_kernel<<<grid, reduce_block, 0, compute_stream()>>>(matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "row sum kernel launch");
    output.mark_host_stale();
    return output;
}

Matrix col_sum(const Matrix& matrix) {
    Matrix output(1, matrix.cols());
    const unsigned grid = matrix.cols() == 0 ? 1 : static_cast<unsigned>((matrix.cols() + 255) / 256);
    col_sum_kernel<<<grid, 256, 0, compute_stream()>>>(matrix.device_data(), output.device_data(), matrix.rows(), matrix.cols());
    checkCuda(cudaGetLastError(), "col sum kernel launch");
    output.mark_host_stale();
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
float Matrix::frobenius_norm() const { return matrix_pro::frobenius_norm(*this); }
float Matrix::abs_max() const { return matrix_pro::abs_max(*this); }
float Matrix::trace() const { return matrix_pro::trace(*this); }
Matrix Matrix::row_sum() const { return matrix_pro::row_sum(*this); }
Matrix Matrix::col_sum() const { return matrix_pro::col_sum(*this); }

float condition_number(const Matrix& matrix) {
    if (matrix.empty()) return 1.0f;
    const auto decomposition = svd(matrix);
    const std::size_t minmn = std::min(matrix.rows(), matrix.cols());
    float max_sigma = 0.0f;
    float min_sigma = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < minmn; ++i) {
        const float sigma = decomposition.s.data()[i * minmn + i];
        max_sigma = std::max(max_sigma, sigma);
        min_sigma = std::min(min_sigma, sigma);
    }
    if (!std::isfinite(min_sigma) || min_sigma <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    return max_sigma / min_sigma;
}

Matrix covariance(const Matrix& matrix) {
    const std::size_t rows = matrix.rows();
    const std::size_t cols = matrix.cols();
    if (rows == 0 || cols == 0) return Matrix(rows, cols);

    Matrix means = col_sum(matrix) * (1.0f / static_cast<float>(rows));
    Matrix centered(rows, cols);
    center_columns_kernel<<<static_cast<unsigned>((matrix.size() + 255) / 256), 256, 0, compute_stream()>>>(
        matrix.device_data(), means.device_data(), centered.device_data(), rows, cols);
    checkCuda(cudaGetLastError(), "center columns kernel launch");

    Matrix result(cols, cols);
    const float alpha = 1.0f / static_cast<float>(rows > 1 ? rows - 1 : 1);
    const float beta = 0.0f;
    const auto covariance_status = cublasSgemm(
        cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
        static_cast<int>(cols), static_cast<int>(cols), static_cast<int>(rows),
        &alpha, centered.device_data(), static_cast<int>(cols),
        centered.device_data(), static_cast<int>(cols),
        &beta, result.device_data(), static_cast<int>(cols));
    if (covariance_status != CUBLAS_STATUS_SUCCESS) throw CudaError("covariance cuBLAS GEMM failed");
    result.mark_host_stale();
    return result;
}

Matrix correlation(const Matrix& matrix) {
    Matrix cov = covariance(matrix);
    const std::size_t dims = cov.rows();
    Matrix result(dims, dims);
    if (dims == 0) return result;

    covariance_to_correlation_kernel<<<static_cast<unsigned>((dims * dims + 255) / 256), 256, 0, compute_stream()>>>(
        cov.device_data(), result.device_data(), dims);
    checkCuda(cudaGetLastError(), "correlation kernel launch");
    result.mark_host_stale();
    return result;
}

float Matrix::condition_number() const { return matrix_pro::condition_number(*this); }
Matrix Matrix::covariance() const { return matrix_pro::covariance(*this); }
Matrix Matrix::correlation() const { return matrix_pro::correlation(*this); }

}
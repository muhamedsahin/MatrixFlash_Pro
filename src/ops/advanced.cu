#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/core/cuda_utils.hpp"

#include <cusolverDn.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace matrix_pro {
namespace {

inline void checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw SolverError(std::string(operation) + ": cusolver status " + std::to_string(static_cast<int>(status)));
    }
}

// RAII guard: frees a CUDA buffer on destruction, even when an exception
// escapes. Zero runtime cost when the guard goes out of scope normally
// (single pointer null-check + branch). Replaces the manual free+throw
// blocks that leaked on intermediate checkCusolver/checkCuda failures.
struct DeviceHold {
    float* ptr;
    explicit DeviceHold(float* p) noexcept : ptr(p) {}
    ~DeviceHold() { if (ptr) free_device_memory(ptr); }
    DeviceHold(const DeviceHold&) = delete;
    DeviceHold& operator=(const DeviceHold&) = delete;
    [[maybe_unused]] float* release() noexcept { float* tmp = ptr; ptr = nullptr; return tmp; }
    float* get() const noexcept { return ptr; }
};
struct IntHolder {
    int* ptr;
    explicit IntHolder(int* p) noexcept : ptr(p) {}
    ~IntHolder() { if (ptr) free_device_memory(ptr); }
    IntHolder(const IntHolder&) = delete;
    IntHolder& operator=(const IntHolder&) = delete;
    [[maybe_unused]] int* release() noexcept { int* tmp = ptr; ptr = nullptr; return tmp; }
    int* get() const noexcept { return ptr; }
};

__global__ void row_major_to_column_major_kernel(const float* in, float* out,
                                                std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const auto row = index / cols;
    const auto col = index % cols;
    out[col * rows + row] = in[row * cols + col];
}

__global__ void column_major_to_row_major_kernel(const float* in, float* out,
                                                std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = rows * cols;
    if (index >= total) return;
    const auto row = index / cols;
    const auto col = index % cols;
    out[row * cols + col] = in[col * rows + row];
}

}

float determinant(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) throw InvalidArgumentError("Determinant requires a square matrix");
    const std::size_t n = matrix.rows();
    if (n == 0) return 1.0f;

    // RAII so every checkCusolver/checkCuda failure path auto-frees.
    DeviceHold device_matrix(static_cast<float*>(allocate_device_memory(n * n * sizeof(float))));
    IntHolder device_ipiv(static_cast<int*>(allocate_device_memory(n * sizeof(int))));
    IntHolder device_info(static_cast<int*>(allocate_device_memory(sizeof(int))));
    float* device_work = nullptr;

    const unsigned blocks = static_cast<unsigned>((n * n + 255) / 256);
    row_major_to_column_major_kernel<<<blocks, 256, 0, compute_stream()>>>(matrix.device_data(), device_matrix.get(), n, n);
    checkCuda(cudaGetLastError(), "determinant conversion kernel launch");

    int lwork = 0;
    checkCusolver(cusolverDnSgetrf_bufferSize(cusolver_handle(), static_cast<int>(n), static_cast<int>(n),
                                             device_matrix.get(), static_cast<int>(n), &lwork), "determinant workspace query");
    {
        DeviceHold work_hold(static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float))));
        device_work = work_hold.get();
        checkCusolver(cusolverDnSgetrf(cusolver_handle(), static_cast<int>(n), static_cast<int>(n), device_matrix.get(),
                                       static_cast<int>(n), device_work, device_ipiv.get(), device_info.get()), "determinant LU");

        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info.get(), sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "determinant info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "determinant info sync");
        if (info != 0) return 0.0f;  // singular -> RAII frees everything, returns 0.

        std::vector<float> host_matrix(n * n);
        std::vector<int> host_ipiv(n);
        checkCuda(cudaMemcpyAsync(host_matrix.data(), device_matrix.get(), n * n * sizeof(float), cudaMemcpyDeviceToHost, compute_stream()), "determinant matrix read");
        checkCuda(cudaMemcpyAsync(host_ipiv.data(), device_ipiv.get(), n * sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "determinant pivot read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "determinant result sync");

        float result = 1.0f;
        int sign = 1;
        for (std::size_t i = 0; i < n; ++i) {
            result *= host_matrix[i * n + i];
            if (host_ipiv[i] != static_cast<int>(i) + 1) sign *= -1;
        }
        return result * static_cast<float>(sign);  // work_hold/device_matrix/ipiv/info freed here.
    }
}

Matrix inverse(const Matrix& matrix) {
    if (matrix.rows() != matrix.cols()) throw InvalidArgumentError("Inverse requires a square matrix");
    const std::size_t n = matrix.rows();
    if (n == 0) return Matrix(0, 0);

    DeviceHold device_matrix(static_cast<float*>(allocate_device_memory(n * n * sizeof(float))));
    DeviceHold device_rhs(static_cast<float*>(allocate_device_memory(n * n * sizeof(float))));
    IntHolder device_ipiv(static_cast<int*>(allocate_device_memory(n * sizeof(int))));
    IntHolder device_info(static_cast<int*>(allocate_device_memory(sizeof(int))));
    float* device_work = nullptr;

    const unsigned blocks = static_cast<unsigned>((n * n + 255) / 256);
    row_major_to_column_major_kernel<<<blocks, 256, 0, compute_stream()>>>(matrix.device_data(), device_matrix.get(), n, n);
    checkCuda(cudaGetLastError(), "inverse conversion kernel launch");

    const float one = 1.0f;
    checkCuda(cudaMemsetAsync(device_rhs.get(), 0, n * n * sizeof(float), compute_stream()), "inverse rhs zero fill");
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t offset = j * n + j;
        checkCuda(cudaMemcpyAsync(device_rhs.get() + offset, &one, sizeof(float), cudaMemcpyHostToDevice, compute_stream()), "inverse rhs identity fill");
    }

    int lwork = 0;
    checkCusolver(cusolverDnSgetrf_bufferSize(cusolver_handle(), static_cast<int>(n), static_cast<int>(n),
                                             device_matrix.get(), static_cast<int>(n), &lwork), "inverse workspace query");
    {
        DeviceHold work_hold(static_cast<float*>(allocate_device_memory(static_cast<std::size_t>(lwork) * sizeof(float))));
        device_work = work_hold.get();
        checkCusolver(cusolverDnSgetrf(cusolver_handle(), static_cast<int>(n), static_cast<int>(n), device_matrix.get(),
                                       static_cast<int>(n), device_work, device_ipiv.get(), device_info.get()), "inverse LU");

        int info = 0;
        checkCuda(cudaMemcpyAsync(&info, device_info.get(), sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "inverse info read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "inverse info sync");
        if (info != 0) throw SolverError("Matrix is singular");

        checkCusolver(cusolverDnSgetrs(cusolver_handle(), CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n),
                                      device_matrix.get(), static_cast<int>(n), device_ipiv.get(),
                                      device_rhs.get(), static_cast<int>(n), device_info.get()), "inverse solve");
        checkCuda(cudaMemcpyAsync(&info, device_info.get(), sizeof(int), cudaMemcpyDeviceToHost, compute_stream()), "inverse info read after solve");
        checkCuda(cudaStreamSynchronize(compute_stream()), "inverse solve info sync");
        if (info != 0) throw SolverError("Matrix inverse failed");
    }

    Matrix output(n, n);
    column_major_to_row_major_kernel<<<blocks, 256, 0, compute_stream()>>>(device_rhs.get(), output.device_data(), n, n);
    checkCuda(cudaGetLastError(), "inverse conversion back kernel launch");
    output.mark_host_stale();  // GPU wrote device_data -> host mirror not yet refreshed.
    return output;
}

float Matrix::determinant() const { return matrix_pro::determinant(*this); }
Matrix Matrix::inverse() const { return matrix_pro::inverse(*this); }

}
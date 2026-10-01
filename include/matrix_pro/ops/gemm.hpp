#pragma once

#include "matrix_pro/core/matrix.hpp"
#include <cstddef>
#include <memory>

namespace matrix_pro {
namespace detail { struct GemmPlanAccess; }

// fp32 forbids reduced-precision input conversion. tf32 explicitly permits it
// on supported GPUs (FP32 storage/accumulation, fewer input mantissa bits).
enum class GemmPrecision { fp32, tf32 };
enum class GemmEpilogue { none, bias, bias_relu, bias_gelu };

struct GemmOptions {
    bool transpose_left = false;
    bool transpose_right = false;
    GemmPrecision precision = GemmPrecision::fp32;
    GemmEpilogue epilogue = GemmEpilogue::none;
    // Limit passed to cuBLASLt algorithm selection/execution. Execution contexts
    // share a lazy 64 MiB reservation per thread/device/stream, not per plan.
    std::size_t workspace_limit_bytes = 64u << 20;
};

struct GemmTuningResult {
    int candidates_tested = 0;
    float selected_ms = 0;
};

// Reusable C = epilogue(alpha * op(A) * op(B) + beta * C).
// Construct on the intended device/thread/compute stream BEFORE graph capture.
// Warm execute performs no allocation, descriptor creation or heuristic query.
// Inputs/output must be distinct contiguous Matrix buffers on that device.
// The plan is bound to its creating thread and stream; never execute it
// concurrently. Buffers and plan must outlive captured graphs using them.
class GemmPlan {
public:
    GemmPlan(std::size_t m, std::size_t n, std::size_t k, GemmOptions options = {});
    ~GemmPlan();
    GemmPlan(GemmPlan&&) noexcept;
    GemmPlan& operator=(GemmPlan&&) noexcept;
    GemmPlan(const GemmPlan&) = delete;
    GemmPlan& operator=(const GemmPlan&) = delete;

    void execute(const Matrix& left, const Matrix& right, Matrix& output,
                 float alpha = 1, float beta = 0, const Matrix* bias = nullptr);
    // Measures available cuBLASLt heuristics. Uses separate scratch output so
    // output and beta's input remain unchanged. Synchronizes, never capture it.
    GemmTuningResult tune(const Matrix& left, const Matrix& right, const Matrix& output,
                          float alpha = 1, float beta = 0, const Matrix* bias = nullptr,
                          int repeats = 5);
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t inner() const noexcept;
    int candidate_count() const noexcept;
    bool uses_cublaslt() const noexcept;

private:
    friend struct detail::GemmPlanAccess;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience APIs use a bounded per-thread/device/stream plan cache.
// Set beta != 0 only when output already has valid DEVICE contents.
void gemm_into(const Matrix& left, const Matrix& right, Matrix& output,
               GemmOptions options = {}, float alpha = 1, float beta = 0,
               const Matrix* bias = nullptr);
Matrix gemm(const Matrix& left, const Matrix& right, GemmOptions options = {},
            float alpha = 1, const Matrix* bias = nullptr);

} // namespace matrix_pro

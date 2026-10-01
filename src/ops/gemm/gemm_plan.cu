// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
// Reusable GEMM plan.
//
// A GemmPlan owns the cuBLASLt matmul descriptor, matrix layouts and the
// chosen algorithm for one (M, N, K, transpose, precision, epilogue, workspace
// limit) problem. Warm execute() does not allocate, does not create
// descriptors and does not query heuristics.
//
// Environment is one per (host thread, CUDA device, compute stream). It holds
// the cuBLAS and cuBLASLt handles plus a single lazy 64 MiB workspace.
// Independent streams never share that scratch. workspace_limit_bytes on
// GemmOptions only caps what the algorithm may use; it does not shrink the
// reservation.
//
// The convenience cache behind gemm() / gemm_into() / gemm_cached_raw() is
// thread_local, bounded to 128 plans, and keyed by device, stream and the
// full option set. Evicting a plan invalidates any CUDA graph that captured
// it, so long-lived graphs must keep an explicit GemmPlan alive.
//
// tune() times at most 16 Lt heuristic candidates into a scratch matrix so
// the caller's C and its beta input stay unchanged. It synchronizes and must
// not run during graph capture. Three short trials, median of the middle
// sample, pick the fastest. Tuning does not speed up every shape: on small
// problems the extra algorithm can be slower than the first heuristic.
//
// Precision: GemmPrecision::fp32 is CUBLAS_COMPUTE_32F_PEDANTIC (no reduced
// input conversion). GemmPrecision::tf32 is CUBLAS_COMPUTE_32F_FAST_TF32
// (Tensor Core, fewer input mantissa bits, FP32 storage and accumulation).
// The legacy raw dispatcher asks for TF32 except for vector GEMMs with no
// epilogue, which stay FP32 so the 1×K×N speedup does not change the old
// numerical contract.

#include "matrix_pro/ops/gemm.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace matrix_pro {
namespace {
#include "plan/environment.inc"
#include "plan/epilogue.cuh"
}

#include "plan/implementation.inc"
#include "plan/public_api.inc"
#include "plan/tuning.inc"
#include "plan/cache.inc"
#include "plan/convenience.inc"
} // namespace matrix_pro

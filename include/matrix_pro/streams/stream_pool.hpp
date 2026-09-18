#pragma once

// Stream pool + async reduction API for maximum overlap.
//
// The default path (compute_stream()) is unchanged: one non-blocking stream
// per (thread, device). The pool below adds N extra streams per device so
// independent operations (e.g. two weight updates) can run concurrently.
// Async argmax/argmin post the result into a caller-owned host-pinned slot and
// invoke the callback from a stream callback -- no pipeline stall.

#include <cstddef>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

namespace matrix_pro {

class Matrix;

// --- stream pool -------------------------------------------------------------
constexpr int kStreamPoolSize = 4;

// Returns pool stream `slot % kStreamPoolSize` for the current device,
// creating it lazily. Never returns compute_stream().
cudaStream_t pool_stream(int slot);
// Blocks the calling thread until every pool stream of the current device is
// idle. Does not touch compute_stream().
void synchronize_pool();
// Runs `fn(stream)` on a pool stream and records `done` when finished; the
// caller can wait on `done` instead of blocking the whole device.
void launch_on_pool(int slot, const std::function<void(cudaStream_t)>& fn,
                    cudaEvent_t done);
// Helper: create/destroy events bound to the pool workflow.
cudaEvent_t make_event();
void destroy_event(cudaEvent_t event);

// --- async reductions ----------------------------------------------------------
// Posts argmax/argmin into *out (must point to host-pinned or page-locked
// memory -- see pin_host()) without blocking the caller; `callback(*out)` is
// invoked from a CUDA stream callback once the value is ready.
void argmax_async(const Matrix& matrix, std::size_t* out,
                  const std::function<void(std::size_t)>& callback = nullptr);
void argmin_async(const Matrix& matrix, std::size_t* out,
                  const std::function<void(std::size_t)>& callback = nullptr);
// Allocate/free page-locked host memory for async result slots.
void* pin_host(std::size_t bytes);
void unpin_host(void* ptr);

} // namespace matrix_pro

#pragma once

// Lightweight dtype abstraction over the float32 core.
// The compute kernels stay float32 (cuBLAS/cuSOLVER fast paths), while this
// header adds:
//  - DType enum + byte-size helpers,
//  - TypedBuffer<T>: RAII device buffer for double/half/float,
//  - fp64 reference reductions (sum/mean/norm via double accumulation),
//  - fp16 storage helpers (pack/unpack float <-> __half on the device).
// Native fp64/fp16 compute matrices are exposed through MatrixD (below) for
// precision-sensitive linalg (SVD/inverse on tiny/large magnitudes).

#include <cstddef>
#include <cstdint>

namespace matrix_pro {

enum class DType : std::uint8_t { f32 = 0, f64 = 1, f16 = 2, bf16 = 3 };

inline constexpr std::size_t dtype_size(DType dt) noexcept {
    return (dt == DType::f64) ? 8u : ((dt == DType::f32) ? 4u : 2u);
}
inline const char* dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::f32: return "f32";
        case DType::f64: return "f64";
        case DType::f16: return "f16";
        case DType::bf16: return "bf16";
    }
    return "unknown";
}

// Owning device buffer of N elements of type T (no host mirror).
template <typename T>
class TypedBuffer {
public:
    TypedBuffer() = default;
    explicit TypedBuffer(std::size_t count);
    ~TypedBuffer();
    TypedBuffer(const TypedBuffer&) = delete;
    TypedBuffer& operator=(const TypedBuffer&) = delete;
    TypedBuffer(TypedBuffer&& other) noexcept;
    TypedBuffer& operator=(TypedBuffer&& other) noexcept;

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class Matrix;

// --- fp64 reference reductions (double accumulation, float storage) ---------
double sum_f64(const Matrix& m);
double mean_f64(const Matrix& m);
double l2_norm_f64(const Matrix& m);

// --- fp16 storage helpers ----------------------------------------------------
// Packs float device data into __half device memory (caller allocates
// count*sizeof(__half) bytes); unpack does the reverse.
void pack_f16(const float* src, void* dst_half, std::size_t count);
void unpack_f16(const void* src_half, float* dst, std::size_t count);

} // namespace matrix_pro

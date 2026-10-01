// Compilation entry point; implementation parts are private to this file.
// Keep this include order and translation unit to preserve inlining,
// kernel launch paths and shared caches. See the adjacent detail.txt.
#include "matrix_pro/core/serialization.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include <fstream>
#include <cstring>
#include <cstdint>

namespace matrix_pro {

// File format magic numbers
constexpr uint32_t MFT_MAGIC = 0x5054464D; // 'M','F','T','P'
constexpr uint32_t MFC_MAGIC = 0x4B43464D; // 'M','F','C','K'

#include "serialization/tensor_io.inc"
#include "serialization/checkpoint_write.inc"
#include "serialization/checkpoint_read.inc"
#include "serialization/checkpoint_access.inc"

} // namespace matrix_pro

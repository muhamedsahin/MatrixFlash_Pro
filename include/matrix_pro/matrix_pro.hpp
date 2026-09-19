#pragma once

// ============================================================================
//  MatrixFlash-Pro — single-include public entry point
// ============================================================================
//
//  #include "matrix_pro/matrix_pro.hpp"
//
//  pulls in every public module of the library:
//    core/      Matrix, Tensor, error hierarchy, memory modes, dtypes, CUDA utils,
//               backend (device info, multi-GPU), memory pool, serialization
//    ops/       elementwise, math (trig, erf, rsqrt...), products, reductions,
//               broadcast, shape, transforms, masking, precision, advanced &
//               cuSOLVER linear algebra, extended linalg (LU, trsm, matrix_exp)
//    nn/        activations, losses, normalization/dropout, conv/pooling,
//               batched GEMM, fused chains, in-place variants
//    autograd/  Variable / VarTensor reverse-mode engine (NoGrad, checkpoint)
//    sparse/    CSR/COO/CSC sparse matrices (spmv, sparse-dense/sparse matmul)
//    indexing/  gather / scatter / embedding primitives
//    view/      zero-copy strided MatrixView
//    rng/       counter-based device RNG (extended distributions, RNGState)
//    streams/   stream pool + async reductions + CUDA Graph + Pipeline
//
//  NOTE: the device-only helpers in matrix_pro/detail/*.cuh define __global__
//  kernels and are deliberately NOT part of this umbrella -- include them from
//  a .cu translation unit only.
//
//  Consumers need the CUDA toolkit include paths (propagated automatically by
//  the CMake target `matrix_pro`).

#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/dtype.hpp"
#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/memory_mode.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/memory_pool.hpp"
#include "matrix_pro/core/backend.hpp"
#include "matrix_pro/core/serialization.hpp"

#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/ops/math.hpp"
#include "matrix_pro/ops/linalg_extended.hpp"

#include "matrix_pro/nn/activation.hpp"
#include "matrix_pro/nn/batch.hpp"
#include "matrix_pro/nn/conv.hpp"
#include "matrix_pro/nn/fused.hpp"
#include "matrix_pro/nn/inplace.hpp"
#include "matrix_pro/nn/loss.hpp"
#include "matrix_pro/nn/ml.hpp"

#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/indexing/indexing.hpp"
#include "matrix_pro/rng/rng.hpp"
#include "matrix_pro/sparse/sparse.hpp"
#include "matrix_pro/sparse/sparse_formats.hpp"
#include "matrix_pro/streams/stream_pool.hpp"
#include "matrix_pro/streams/execution.hpp"
#include "matrix_pro/view/view.hpp"


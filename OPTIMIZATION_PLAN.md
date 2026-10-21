# MatrixFlash-Pro performance and capability roadmap

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

## Current kernel (evening of 2026-10-21)

Machine: AMD Ryzen 7 5800H. AVX-512 is not present and is not selected.
`gemm_f32` keeps `avx2-4x16` when AVX2 and FMA exist, because the serial
AVX-512 kernel does not distribute rows. `gemm_kernel_name()` on this CPU is
`avx2-4x16`.

Deleted experiments were slower. They are not current kernels:

- an `Nc` tile that repacks A
- MASM 6×16
- hybrid GEMM at `N=512`

The heterogeneous scheduler (CPU rows through `gemm_f32`, GPU rows through
`cublasSgemm`) is still the row-split path. It is not the deleted `N=512`
hybrid GEMM.

Square GEMM, `N=2048`, this evening: Nexus 330, 362 and 369 GFLOPS; TensorFlow
389 GFLOPS. No world ranking. No other GFLOPS are added here.

cuDNN is force-disabled (`tools/build_local.py` passes
`-DMATRIX_PRO_ENABLE_CUDNN=OFF`). Device GEMM is cuBLAS. The CUDA conv kernel
was measured on this machine against Nexus CPU conv (shape `8×3×32×32`, 16
filters, `3×3`, pad 1): CPU 6.39 GFLOPS, device-resident kernel 48.7 GFLOPS,
host round-trip 19.0 GFLOPS. Those conv figures are not compared with PyTorch.

## Earlier GPU plan (2026-10-01)

Date: 2026-10-01. Existing public APIs and host-mirror semantics are preserved.
No global fastest-library claim is justified by a single GPU or GEMM benchmark.
The items below describe the cuBLAS plan work. They do not revive the deleted
CPU kernels above, and they do not select AVX-512 on this CPU.

## Ordered delivery

1. Establish a clean Release build and record baseline correctness/performance.
   Keep pre-existing workspace deletions outside the commits.
2. Replace repeated cuBLASLt descriptor/heuristic construction and global shared
   workspaces with bounded, thread/device/stream-isolated reusable execution plans.
   Reuse the same engine for fused bias/activation operations. Handle empty inner
   dimensions, output aliasing and integer limits explicitly.
3. Add reusable public GEMM plans with transpose flags, alpha/beta accumulation,
   explicit FP32/TF32 policy, fused epilogues, optional measured algorithm tuning,
   and a graph-compatible, preallocated execution path. Add advanced numerical
   operations with mathematical reference tests.
4. Measure preallocated GEMM against raw cuBLAS/cuBLASLt with identical precision,
   nonzero seeded data, correctness validation, warmups, repeated CUDA-event and
   wall-clock samples. Include small, irregular and rectangular cases. Measure
   optional NumPy/PyTorch/CuPy separately without mixing CPU/GPU timing scopes.
5. Run the full behavior suite, report gains and regressions with raw data and
   reproduction commands, then commit each finished stage and push to origin.

## Acceptance criteria

- Existing tests pass on the available RTX 3070 Laptop GPU.
- New operations agree with independent CPU mathematical references.
- Plans reject invalid dimensions, aliasing, and execution on the wrong device.
- Warm execution avoids descriptor creation, heuristic search and allocation.
- Concurrent host threads do not share mutable descriptors or scratch buffers.
- Performance results record hardware, toolkit, policy, timing scope and variance.
- Fast math and reduced precision remain explicit choices; default APIs retain
  their existing numerical policy. Unsupported rivals are reported as skipped.

## Follow-on work requiring additional hardware/evidence

Multi-GPU validation; Hopper/Blackwell-specific kernels; exhaustive autotuning;
distributed algebra; structured/sparse workloads and application-level profiling.
These are future work, not claims about the delivered implementation.

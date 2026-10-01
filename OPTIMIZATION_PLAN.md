# MatrixFlash-Pro performance and capability roadmap

Date: 2026-10-01. Existing public APIs and host-mirror semantics are preserved.
No global fastest-library claim is justified by a single GPU or GEMM benchmark.

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

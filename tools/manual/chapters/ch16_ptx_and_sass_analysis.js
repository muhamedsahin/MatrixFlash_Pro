// tools/manual/chapters/ch16_ptx_and_sass_analysis.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 16</div>
      <h1 class="chapter-title">Low-Level PTX Assembly & SASS Instruction Scheduling</h1>
    </div>

    <h2>16.1 PTX vs SASS: The GPU Compilation Pipeline</h2>
    <p>
      In NVIDIA CUDA compilation, C++ source code is first translated into <strong>Parallel Thread Execution (PTX)</strong>—an architecture-independent low-level virtual instruction set. The CUDA driver JIT or offline compiler (<code>ptxas</code>) then compiles PTX into <strong>SASS (Source-Associated Assembly)</strong>, the true machine code executed directly by the SM instruction issue units:
    </p>

    <div class="arch-diagram">
+---------------------------------------------------------------------------------------------------+
|                         MATRIXFLASH-PRO COMPILATION PIPELINE TO SASS                              |
+---------------------------------------------------------------------------------------------------+
  matrix_pro C++17 Source
             |
             v  (NVCC Frontend with Clang AST parsing)
  PTX Virtual Assembly (e.g., ld.global.v4.f32, fma.rn.f32, shfl.sync.down)
             |
             v  (ptxas optimizing assembler: Register Allocation, Loop Unrolling, Pipe Scheduling)
  SASS Native Machine Code (e.g., LDG.E.128, FFMA, SHF.L, HMMA.16816)
             |
             v
  GPU Hardware Execution on Ampere SM (Zero Pipeline Stalls)
    </div>

    <h2>16.2 Disassembly of Vectorized Memory Pipelines</h2>
    <p>
      Examining the compiled SASS disassembly of MatrixFlash-Pro vectorized element-wise kernels reveals that scalar loops are completely eliminated in favor of single-cycle 128-bit memory instructions:
    </p>

    <pre><code>// SASS Disassembly of MatrixFlash-Pro Vectorized Load/Store Pipeline
// Architecture: sm_86 (Ampere)
.L_LOOP:
  LDG.E.128         R4, [R2.64]            ; Load 16 bytes (float4: x, y, z, w) from Global VRAM
  LDG.E.128         R8, [R0.64]            ; Load 16 bytes from second tensor operand
  FFMA              R12, R4, R8, R16       ; Vectorized Fused Multiply-Add (Lane 0)
  FFMA              R13, R5, R9, R17       ; Vectorized Fused Multiply-Add (Lane 1)
  FFMA              R14, R6, R10, R18      ; Vectorized Fused Multiply-Add (Lane 2)
  FFMA              R15, R7, R11, R19      ; Vectorized Fused Multiply-Add (Lane 3)
  STG.E.128         [R18.64], R12          ; Store 16 bytes directly into Output VRAM
  IADD3             R2, R2, 0x10, RZ       ; Advance memory pointers by 16 bytes
  IADD3             R0, R0, 0x10, RZ
  ISETP.LT.AND      P0, PT, R2, R20, PT    ; Check loop bound in integer ALU
  @P0 BRA           .L_LOOP                ; Branch to next unrolled iteration</code></pre>

    <div class="page-subbreak"></div>

    <h2>16.3 Warp Scheduler Dual-Issue Optimization</h2>
    <p>
      On Ampere (sm_86) and Ada Lovelace (sm_89) architectures, each SM sub-partition features dual dispatch units capable of issuing two instructions per cycle per warp, provided the instructions target separate execution pipelines:
    </p>

    <div class="formula-box">
      $$\text{Instruction Level Parallelism (ILP)}: \quad \text{Dispatch Lane 0} = \text{FP32 ALU (FFMA)} \quad \parallel \quad \text{Dispatch Lane 1} = \text{LSU / Branch / INT32}$$
      <span class="eq-desc">MatrixFlash-Pro interleaves arithmetic with pointer increments to saturate dual-issue units</span>
    </div>

    <p>
      By manually tuning loop unroll depths (<code>#pragma unroll 4</code>), MatrixFlash-Pro ensures that memory loads are hoisted ahead of arithmetic, providing sufficient instruction distance to completely hide the 400-cycle global memory latency without spilling registers to local stack memory.
    </p>
  </div>
`;


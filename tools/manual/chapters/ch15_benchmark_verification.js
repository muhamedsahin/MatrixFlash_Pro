// tools/manual/chapters/ch15_benchmark_verification.js

module.exports = `
  <div class="chapter">
    <div class="chapter-header">
      <div class="chapter-num">Chapter 15</div>
      <h1 class="chapter-title">Fair GEMM Comparison, 1 October 2026</h1>
    </div>

    <h2>15.1 Hardware and Protocol</h2>
    <p>
      Measurements were taken on one laptop. Clocks and the power limit were not locked. A few microseconds on a small shape are not a ranking.
    </p>
    <table>
      <thead>
        <tr><th>Field</th><th>Value</th></tr>
      </thead>
      <tbody>
        <tr><td>GPU</td><td>NVIDIA GeForce RTX 3070 Laptop, 8 GiB, SM 86</td></tr>
        <tr><td>CPU</td><td>AMD Ryzen 7 5800H</td></tr>
        <tr><td>Software</td><td>Windows 11 WDDM, driver 610.88, CUDA 13.4, MSVC 14.44, Release, native SM 86, cuDNN off</td></tr>
        <tr><td>GPU timer</td><td>CUDA events. Two runs, 10 warmups, 30 repeats, batch 32. Tables use the median of 60 samples.</td></tr>
        <tr><td>What is excluded</td><td>Allocation, upload, download, descriptor creation, heuristic search and tuning</td></tr>
        <tr><td>Rivals</td><td>Raw cuBLAS and raw cuBLASLt reuse descriptors and the same 64 MiB workspace. They are not started cold.</td></tr>
        <tr><td>Not measured</td><td>PyTorch and CuPy were not installed. JAX, ArrayFire and Eigen were not run. No 4096² case in this protocol.</td></tr>
      </tbody>
    </table>
    <p>
      FP32 means <code>CUBLAS_COMPUTE_32F_PEDANTIC</code>. TF32 means <code>CUBLAS_COMPUTE_32F_FAST_TF32</code> (fewer input mantissa bits, FP32 storage and accumulation). Do not rank those two tables against each other as if the answers had the same accuracy. The largest sampled absolute error in the GPU runs was 1.12e-5 for FP32 and 3.31e-3 for TF32, on the sampled entries of those inputs, not as a general bound.
    </p>
    <p>
      An earlier September protocol reported 16,895 GFLOPS at 4096². That run used 12 warmups and 40 repeats and is not part of this chapter.
    </p>

    <h2>15.2 Library Default, Before and After</h2>
    <p>
      Ratio = old median / new median. Below 1 is a regression. The default policy is shape-dependent, so this table is not a same-precision vendor comparison. Times are microseconds.
    </p>
    <table>
      <thead>
        <tr><th>Shape</th><th>Old</th><th>New</th><th>Old/new</th></tr>
      </thead>
      <tbody>
        <tr><td>512³</td><td>34.83</td><td>31.34</td><td>1.11×</td></tr>
        <tr><td>1024³</td><td>159.4</td><td>187.6</td><td>0.85× regression</td></tr>
        <tr><td>2048³</td><td>1165</td><td>1289</td><td>0.90× regression</td></tr>
        <tr><td>127×257×63</td><td>26.24</td><td>13.86</td><td>1.89×</td></tr>
        <tr><td>64×1024×512</td><td>36.75</td><td>23.86</td><td>1.54×</td></tr>
        <tr><td>1×1024×4096</td><td>185.8</td><td>46.66</td><td><strong>3.98×</strong>, FP32 kept</td></tr>
        <tr><td>1024×1×4096</td><td>42.37</td><td>42.98</td><td>0.99×</td></tr>
      </tbody>
    </table>
    <p>
      The long vector-times-matrix path moved to cuBLASLt without dropping to TF32. That is the clearest gain against the previous build. The new default is slower on the large squares; those rows stay in the table.
    </p>

    <h2>15.3 Strict FP32 versus Raw cuBLAS and cuBLASLt</h2>
    <p>Microseconds and GFLOPS (2·M·N·K / time). The tuned plan has already selected among at most 16 heuristics.</p>
    <table>
      <thead>
        <tr><th>Shape</th><th>Plan</th><th>Tuned plan</th><th>Raw cuBLAS</th><th>Raw cuBLASLt</th></tr>
      </thead>
      <tbody>
        <tr><td>512³</td><td>43.6 µs · 6159</td><td>45.9 µs · 5846</td><td>48.0 µs · 5589</td><td>48.4 µs · 5548</td></tr>
        <tr><td>1024³</td><td>365 µs · 5886</td><td><strong>277 µs · 7761</strong></td><td>343 µs · 6253</td><td>347 µs · 6182</td></tr>
        <tr><td>2048³</td><td>2339 µs · 7343</td><td><strong>2041 µs · 8419</strong></td><td>2389 µs · 7192</td><td>2365 µs · 7266</td></tr>
        <tr><td>64×1024×512</td><td>25.0 µs</td><td><strong>16.0 µs</strong></td><td>34.5 µs</td><td>28.3 µs</td></tr>
        <tr><td>1×1024×4096</td><td>45.4 µs</td><td>48.3 µs</td><td>45.5 µs</td><td>45.3 µs</td></tr>
      </tbody>
    </table>
    <p>
      At 1024³ the tuned plan is about 1.24× the raw cuBLAS time (343 / 277). At 2048³ it is about 1.17× (2389 / 2041). On 16³–64³ the launch cost dominates the arithmetic, so a GFLOPS order is not a kernel ranking.
    </p>

    <h2>15.4 TF32 Tensor Cores</h2>
    <table>
      <thead>
        <tr><th>Shape</th><th>Plan</th><th>Tuned plan</th><th>Raw cuBLAS</th><th>Raw cuBLASLt</th></tr>
      </thead>
      <tbody>
        <tr><td>512³</td><td>36.1 µs · 7430</td><td>27.8 µs · 9659</td><td>35.2 µs · 7626</td><td>34.8 µs · 7703</td></tr>
        <tr><td>1024³</td><td>186 µs · 11530</td><td>176 µs · 12185</td><td><strong>169 µs · 12694</strong></td><td>171 µs · 12524</td></tr>
        <tr><td>2048³</td><td>1321 µs · 13010</td><td>1307 µs · 13142</td><td>1325 µs · 12967</td><td><strong>1304 µs · 13174</strong></td></tr>
        <tr><td>1024×64×512</td><td>23.8 µs</td><td><strong>16.3 µs</strong></td><td>26.2 µs</td><td>32.8 µs</td></tr>
        <tr><td>1×1024×4096</td><td>46.6 µs</td><td>46.7 µs</td><td>45.2 µs</td><td>45.5 µs</td></tr>
      </tbody>
    </table>
    <p>
      At TF32 1024³ raw cuBLAS is the fastest. The tuned plan is about 96% of that rate. At 2048³ the tuned plan and raw Lt are the same band; 13.1 TFLOPS is the peak of this suite. On 1024×64×512 the tuned plan is clearly faster than raw Lt. The ranking is not uniform.
    </p>

    <h2>15.5 NumPy / OpenBLAS</h2>
    <p>
      NumPy 2.3.5 with OpenBLAS 0.3.30, wall-clock, CPU thread count not pinned. The GPU column is the strict FP32 plan wall time, including the wait at the end of the batch. No H2D/D2H transfer is included. The timers are different. Below 64³ the CPU is faster because GPU launch overhead exceeds the arithmetic.
    </p>
    <table>
      <thead>
        <tr><th>Shape</th><th>GPU FP32 plan wall</th><th>NumPy wall</th><th>CPU / GPU</th></tr>
      </thead>
      <tbody>
        <tr><td>64³</td><td>0.016 ms</td><td>0.0099 ms</td><td>CPU faster</td></tr>
        <tr><td>256³</td><td>0.015 ms</td><td>0.261 ms</td><td>17×</td></tr>
        <tr><td>512³</td><td>0.045 ms</td><td>1.094 ms</td><td>25×</td></tr>
        <tr><td>1024³</td><td>0.367 ms</td><td>4.910 ms</td><td>13.4×</td></tr>
        <tr><td>2048³</td><td>2.341 ms</td><td>41.16 ms</td><td>17.6×</td></tr>
      </tbody>
    </table>

    <h2>15.6 CUDA Graph</h2>
    <p>
      One graph captures 32 GEMMs and the replay time is divided by 32. This is not 32 separate launches. The graph amortizes host submission on small and medium problems. An application that reads every output on the host does not see the same gain.
    </p>
    <table>
      <thead>
        <tr><th>Shape</th><th>FP32 plan</th><th>FP32 raw Lt</th><th>TF32 plan</th><th>TF32 raw Lt</th></tr>
      </thead>
      <tbody>
        <tr><td>128³</td><td>4.93 µs</td><td>4.83 µs</td><td>4.90 µs</td><td>4.93 µs</td></tr>
        <tr><td>512³</td><td>45.2 µs</td><td>47.1 µs</td><td>24.7 µs</td><td>31.8 µs</td></tr>
        <tr><td>1024³</td><td>269 µs</td><td>344 µs</td><td>166 µs</td><td>167 µs</td></tr>
        <tr><td>2048³</td><td>2040 µs</td><td>2386 µs</td><td>1318 µs</td><td>1313 µs</td></tr>
      </tbody>
    </table>

    <h2>15.7 What This Chapter Does Not Claim</h2>
    <ul>
      <li>No fused-epilogue, LU-reuse, sparse or <code>matrix_exp</code> speed table was produced in this round.</li>
      <li>No PyTorch, CuPy, JAX, ArrayFire or Eigen timing.</li>
      <li>No locked-clock roofline and no “98% of silicon” statement. The previous edition of this chapter made that claim from a different protocol; it is withdrawn here.</li>
      <li>Single-laptop WDDM numbers do not establish a universal ranking.</li>
    </ul>
    <p>
      Reproduce with <code>matrix_pro_bench_gemm_fair</code> and <code>tools/bench_rivals_fair.py</code>, warmup 10, repeats 30, batch 32, two GPU runs. Raw JSON is under <code>benchmarks/reports/2026-10-01/</code>.
    </p>
  </div>
`;

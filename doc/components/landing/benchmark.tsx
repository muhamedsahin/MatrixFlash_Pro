'use client'

import { Reveal } from '@/components/reveal'
import { CodeBlock } from '@/components/code-block'
import { useLanguage } from '@/lib/language-context'
import { Gauge, Cpu } from 'lucide-react'

export function Benchmark() {
  const { lang, t } = useLanguage()

  const stats = [
    { value: '~16.8', label: { tr: 'TFLOPS @ 2048² (into)', en: 'TFLOPS @ 2048² (into)' } },
    { value: '99–138%', label: { tr: 'cuBLAS Oranı', en: 'cuBLAS Ratio' } },
    { value: '6×+', label: { tr: 'Fused vs Zincir', en: 'Fused vs Chain' } },
    { value: '~37×', label: { tr: 'vs NumPy OpenBLAS', en: 'vs NumPy OpenBLAS' } },
  ]

  const benchCode = `# GPU + rakip motorlar + doc sync
cmake --build --preset release --target matrix_pro_bench_comparison matrix_pro_bench_external_gemm
./build/benchmarks/Release/matrix_pro_bench_comparison.exe --sizes 256,512,1024,2048 --warmup 12 --repeats 40 --json benchmarks/results/comparison.json
py -3 tools/bench_rivals.py
python tools/gen_bench_page.py
python tools/sync_benchmark_data.py`

  const comparisonData = [
    {
      name: t('NumPy @ OpenBLAS CPU (1024², ölçüldü)', 'NumPy @ OpenBLAS CPU (1024², measured)'),
      time: '6.3 ms / 343 GFLOPS',
      speedup: '1× CPU',
      width: 3,
      color: 'from-zinc-500 to-zinc-400',
    },
    {
      name: t('NVIDIA cublasLt (1024², ölçüldü)', 'NVIDIA cublasLt (1024², measured)'),
      time: '0.28 ms / 7600 GFLOPS',
      speedup: '~22×',
      width: 68,
      color: 'from-violet-500 to-violet-300',
    },
    {
      name: t('Ham cuBLAS (1024², ölçüldü)', 'Raw cuBLAS (1024², measured)'),
      time: '0.19 ms / 11275 GFLOPS',
      speedup: '~33×',
      width: 99,
      color: 'from-cyan-500 to-cyan-300',
    },
    {
      name: t('MatrixFlash-Pro multiply_into (1024²)', 'MatrixFlash-Pro multiply_into (1024²)'),
      time: '0.19 ms / 11155 GFLOPS',
      speedup: '~33× · %99 cuBLAS',
      width: 100,
      color: 'from-emerald-500 to-primary',
    },
  ]

  return (
    <section className="relative border-t border-border/80 py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid gap-14 lg:grid-cols-2 lg:items-center">
          <Reveal>
            <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
              <Gauge className="size-3.5" />
              <span>{t('CUBlAS SINIFINDA — ÖLÇÜLDÜ', 'CUBLAS-CLASS — MEASURED')}</span>
            </div>
            <h2 className="mt-4 text-balance text-3xl font-extrabold tracking-tight sm:text-4xl">
              {t(
                'cuBLAS ile aynı bant — çoğu boyutta eşit veya daha hızlı',
                'Same band as cuBLAS — equal or faster on most sizes',
              )}
            </h2>
            <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
              {t(
                'Algo-cache’li cublasLt + TENSOR_OP ile 1024²’de cuBLAS’ın %99’u, 2048² ve 512²’de geçiyoruz. NumPy@OpenBLAS’a karşı ~37×. PyTorch/CuPy/ArrayFire/JAX de cuBLAS tavanına oturur; fark C++17 API ve fused epilogue.',
                'With algo-cached cublasLt + TENSOR_OP we hit 99% of cuBLAS at 1024² and beat it at 2048² and 512². ~37× over NumPy@OpenBLAS. PyTorch/CuPy/ArrayFire/JAX also sit on the cuBLAS ceiling; the delta is a C++17 API and fused epilogues.',
              )}
            </p>

            <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {stats.map((s, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-border/80 bg-card/60 p-4 shadow-sm backdrop-blur-md"
                >
                  <div className="font-mono text-2xl font-extrabold text-primary text-glow">
                    {s.value}
                  </div>
                  <div className="mt-1 text-[11px] leading-snug text-muted-foreground">
                    {s.label[lang]}
                  </div>
                </div>
              ))}
            </div>
          </Reveal>

          <Reveal delay={100}>
            <div className="rounded-2xl border border-border/80 glass-panel surface-shine p-6 shadow-2xl">
              <div className="flex items-center justify-between border-b border-border/60 pb-3">
                <span className="flex items-center gap-2 font-mono text-xs font-bold text-foreground">
                  <Cpu className="size-4 text-primary" />
                  {t('1024×1024 GEMM (ölçüldü)', '1024×1024 GEMM (measured)')}
                </span>
                <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary">
                  RTX 3070 Laptop · sm_86
                </span>
              </div>

              <div className="mt-5 space-y-4">
                {comparisonData.map((item, idx) => (
                  <div key={idx} className="space-y-1.5">
                    <div className="flex items-center justify-between font-mono text-xs">
                      <span className="max-w-[280px] truncate text-foreground">{item.name}</span>
                      <span className="font-bold text-primary">
                        {item.time} ({item.speedup})
                      </span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary/80">
                      <div
                        className={`h-full rounded-full bg-gradient-to-r ${item.color}`}
                        style={{ width: `${item.width}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 space-y-3 border-t border-border/50 pt-4">
                <CodeBlock code={benchCode} filename="benchmark.ps1" />
                <a
                  href="/docs/performans"
                  className="inline-flex items-center gap-2 font-mono text-xs font-bold text-primary hover:underline"
                >
                  {t(
                    'Detaylı tablo + rakip motorlar: /docs/performans',
                    'Full table + rival engines: /docs/performans',
                  )}
                </a>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

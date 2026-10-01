'use client'

import { Reveal } from '@/components/reveal'
import { CodeBlock } from '@/components/code-block'
import { useLanguage } from '@/lib/language-context'
import { Gauge, Cpu } from 'lucide-react'

export function Benchmark() {
  const { lang, t } = useLanguage()

  const stats = [
    { value: '13.1', label: { tr: 'TFLOPS TF32 @ 2048²', en: 'TFLOPS TF32 @ 2048²' } },
    { value: '1.24×', label: { tr: 'FP32 tuned vs cuBLAS @ 1024²', en: 'FP32 tuned vs cuBLAS @ 1024²' } },
    { value: '3.98×', label: { tr: 'Uzun vektör, eski sürüme göre', en: 'Long vector vs previous build' } },
    { value: '13×', label: { tr: '1024² FP32 plan vs NumPy duvar', en: '1024² FP32 plan vs NumPy wall' } },
  ]

  const benchCode = `python tools/build_local.py --reconfigure --jobs 4
.\\build\\perf-release\\benchmarks\\matrix_pro_bench_gemm_fair.exe --output benchmarks/results/run.json --warmup 10 --repeats 30 --batch 32
python tools/bench_rivals_fair.py --output benchmarks/results/rivals_fair.json --warmup 10 --repeats 30 --batch 32`

  const comparisonData = [
    {
      name: t('NumPy @ OpenBLAS (1024², CPU duvar)', 'NumPy @ OpenBLAS (1024², CPU wall)'),
      time: '4.91 ms',
      speedup: 'CPU',
      width: 4,
      color: 'from-zinc-500 to-zinc-400',
    },
    {
      name: t('MatrixFlash-Pro tuned plan (1024² TF32)', 'MatrixFlash-Pro tuned plan (1024² TF32)'),
      time: '0.176 ms / 12185 GFLOPS',
      speedup: '%96 cuBLAS',
      width: 96,
      color: 'from-emerald-500 to-primary',
    },
    {
      name: t('Ham cuBLASLt (1024² TF32)', 'Raw cuBLASLt (1024² TF32)'),
      time: '0.171 ms / 12524 GFLOPS',
      speedup: '%99',
      width: 99,
      color: 'from-violet-500 to-violet-300',
    },
    {
      name: t('Ham cuBLAS (1024² TF32)', 'Raw cuBLAS (1024² TF32)'),
      time: '0.169 ms / 12694 GFLOPS',
      speedup: t('en hızlı TF32', 'fastest TF32'),
      width: 100,
      color: 'from-cyan-500 to-cyan-300',
    },
  ]

  return (
    <section className="relative border-t border-border/80 py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid gap-14 lg:grid-cols-2 lg:items-center">
          <Reveal>
            <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
              <Gauge className="size-3.5" />
              <span>{t('1 EKİM 2026 — AYNI HASSASİYET', '1 OCTOBER 2026 — SAME PRECISION')}</span>
            </div>
            <h2 className="mt-4 text-balance text-3xl font-extrabold tracking-tight sm:text-4xl">
              {t(
                'TF32 büyük karelerde cuBLAS ile aynı bant',
                'Same band as cuBLAS on large TF32 squares',
              )}
            </h2>
            <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
              {t(
                'RTX 3070 Laptop, önceden ayrılmış tampon. TF32 1024²’de ham cuBLAS 0.169 ms; tuned plan 0.176 ms. FP32 1024²’de tuned plan ham cuBLAS’tan 1.24× hızlı. PyTorch ve CuPy bu turda kurulu değildi.',
                'RTX 3070 Laptop, preallocated buffers. At TF32 1024² raw cuBLAS is 0.169 ms and the tuned plan is 0.176 ms. At FP32 1024² the tuned plan is 1.24× faster than raw cuBLAS. PyTorch and CuPy were not installed this round.',
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
                  {t('1024² TF32 GEMM (ölçüldü)', '1024² TF32 GEMM (measured)')}
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

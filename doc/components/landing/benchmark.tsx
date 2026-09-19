'use client'

import { Reveal } from '@/components/reveal'
import { CodeBlock } from '@/components/code-block'
import { useLanguage } from '@/lib/language-context'
import { Zap, Gauge, Cpu } from 'lucide-react'

export function Benchmark() {
  const { lang, t } = useLanguage()

  const stats = [
    { value: '100+', label: { tr: 'GPU Operasyonu', en: 'GPU Operations' } },
    { value: 'TF32', label: { tr: 'Tensor Core Desteği', en: 'Tensor Core Support' } },
    { value: '0 Byte', label: { tr: 'View Kopyalama Maliyeti', en: 'View Strided Overhead' } },
    { value: '4 Stream', label: { tr: 'Eş Zamanlı İşlem Havuzu', en: 'Concurrent Stream Pool' } },
  ]

  const benchCode = `# Karsilastirma benchmark'ini derle ve calistir
cmake --build --preset release --target matrix_pro_bench_comparison
./build/benchmarks/Release/matrix_pro_bench_comparison.exe --sizes 256,512,1024,2048 --repeats 10 --warmup 3 --csv benchmarks/results/comparison.csv --json benchmarks/results/comparison.json`

  const comparisonData = [
    { name: 'CPU tek-thread naive (256x256, olculdu)', time: '15.02 ms', speedup: '1x (Referans)', width: 6, color: 'from-zinc-500 to-zinc-400' },
    { name: 'Naive GPU global-memory kernel (1024x1024, olculdu)', time: '2.23 ms', speedup: '~50x', width: 30, color: 'from-amber-500 to-amber-400' },
    { name: 'MatrixFlash-Pro operator* (1024x1024, olculdu)', time: '0.75 ms', speedup: '~141x', width: 68, color: 'from-cyan-500 to-cyan-400' },
    { name: 'Ham cuBLAS tahsissiz (1024x1024, olculdu)', time: '0.19 ms', speedup: '~545x', width: 100, color: 'from-emerald-500 to-primary' },
  ]

  return (
    <section className="relative py-24 border-t border-border/80">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid gap-14 lg:grid-cols-2 lg:items-center">
          <Reveal>
            <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
              <Gauge className="size-3.5" />
              <span>{t('DONANIM SINIRINDA PERFORMANS', 'HARDWARE-SATURATING BENCHMARK')}</span>
            </div>
            <h2 className="mt-4 text-balance text-3xl font-extrabold tracking-tight sm:text-4xl">
              {t(
                'GPU Üzerinde Ölçülebilir, Sürdürülebilir Hız',
                'Measurable, Sustained GPU Acceleration',
              )}
            </h2>
            <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
              {t(
                'NVIDIA cuBLAS entegrasyonu, TF32 Tensor Core donanım birimlerini doğrudan devreye sokarak FP32 işlemlerinde kayda değer bir TFLOPS sıçraması sağlar. Fused kernel zincirleri ise ara bellek gecikmelerini sıfırlar.',
                'NVIDIA cuBLAS integration directly utilizes TF32 Tensor Cores for massive TFLOPS throughput, while fused chains eliminate intermediate memory traffic bottlenecks.',
              )}
            </p>

            <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {stats.map((s, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-border/80 bg-card/60 p-4 backdrop-blur-md shadow-sm"
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
            <div className="rounded-2xl border border-border/80 bg-card/50 p-6 shadow-2xl backdrop-blur-xl">
              <div className="flex items-center justify-between border-b border-border/60 pb-3">
                <span className="font-mono text-xs font-bold text-foreground flex items-center gap-2">
                  <Cpu className="size-4 text-primary" />
                  2048 × 2048 Matris Çarpımı (GEMM Benchmark)
                </span>
                <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary">
                  NVIDIA RTX / CUDA 12.3+
                </span>
              </div>

              {/* Comparison Bars */}
              <div className="mt-5 space-y-4">
                {comparisonData.map((item, idx) => (
                  <div key={idx} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-foreground truncate max-w-[280px]">{item.name}</span>
                      <span className="text-primary font-bold">{item.time} ({item.speedup})</span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary/80">
                      <div
                        className={`h-full rounded-full bg-gradient-to-r ${item.color} shadow-[0_0_10px_rgba(110,231,183,0.3)]`}
                        style={{ width: `${item.width}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 pt-4 border-t border-border/50 space-y-3">
                <CodeBlock code={benchCode} filename="benchmark.ps1" />
                <a
                  href="/docs/performans"
                  className="inline-flex items-center gap-2 font-mono text-xs font-bold text-primary hover:underline"
                >
                  Detayli karsilastirma tablosu + interaktif grafik: /docs/performans
                </a>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

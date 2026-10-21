'use client'

import { Reveal } from '@/components/reveal'
import { CodeBlock } from '@/components/code-block'
import { useLanguage } from '@/lib/language-context'
import { Gauge, Cpu } from 'lucide-react'

export function Benchmark() {
  const { lang, t } = useLanguage()

  const stats = [
    { value: '330', label: { tr: 'Nexus GEMM N=2048, GFLOPS', en: 'Nexus GEMM N=2048, GFLOPS' } },
    { value: '362', label: { tr: 'Nexus GEMM N=2048, GFLOPS', en: 'Nexus GEMM N=2048, GFLOPS' } },
    { value: '369', label: { tr: 'Nexus GEMM N=2048, GFLOPS', en: 'Nexus GEMM N=2048, GFLOPS' } },
    { value: '389', label: { tr: 'TensorFlow, aynı N, GFLOPS', en: 'TensorFlow, same N, GFLOPS' } },
  ]

  const benchCode = `python tools/build_local.py --reconfigure --jobs 4`

  const comparisonData = [
    {
      name: t('Nexus GEMM N=2048', 'Nexus GEMM N=2048'),
      time: '330 GFLOPS',
      speedup: 'avx2-4x16',
      width: 85,
      color: 'from-emerald-500 to-primary',
    },
    {
      name: t('Nexus GEMM N=2048', 'Nexus GEMM N=2048'),
      time: '362 GFLOPS',
      speedup: 'avx2-4x16',
      width: 93,
      color: 'from-emerald-500 to-primary',
    },
    {
      name: t('Nexus GEMM N=2048', 'Nexus GEMM N=2048'),
      time: '369 GFLOPS',
      speedup: 'avx2-4x16',
      width: 95,
      color: 'from-emerald-500 to-primary',
    },
    {
      name: 'TensorFlow N=2048',
      time: '389 GFLOPS',
      speedup: t('aynı protokol', 'same protocol'),
      width: 100,
      color: 'from-cyan-500 to-cyan-300',
    },
  ]

  const convData = [
    {
      name: t('Nexus CPU conv', 'Nexus CPU conv'),
      time: '6.39 GFLOPS',
      width: 13,
      color: 'from-zinc-500 to-zinc-400',
    },
    {
      name: t('Host gidiş-dönüş', 'Host round-trip'),
      time: '19.0 GFLOPS',
      width: 39,
      color: 'from-violet-500 to-violet-300',
    },
    {
      name: t('Cihazda duran CUDA çekirdeği', 'Device-resident CUDA kernel'),
      time: '48.7 GFLOPS',
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
              <span>{t('21 EKİM 2026 AKŞAMI — BU MAKİNE', '21 OCTOBER 2026 EVENING — THIS MACHINE')}</span>
            </div>
            <h2 className="mt-4 text-balance text-3xl font-extrabold tracking-tight sm:text-4xl">
              {t(
                'Ryzen 7 5800H, çekirdek avx2-4x16',
                'Ryzen 7 5800H, kernel avx2-4x16',
              )}
            </h2>
            <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
              {t(
                'AVX-512 bu CPU’da yok ve seçilmiyor. GEMM N=2048: Nexus 330, 362 ve 369 GFLOPS; TensorFlow 389 GFLOPS. Silinen denemeler (A’yı yeniden paketleyen Nc, MASM 6×16, N=512 hibrit GEMM) güncel çekirdek değildir. cuDNN zorla kapalıdır. Conv şekli 8×3×32×32, 16 filtre, 3×3, dolgu 1: CPU 6.39, cihazda duran çekirdek 48.7, host gidiş-dönüş 19.0 GFLOPS. Bu conv sayısı PyTorch ile karşılaştırılmaz.',
                'AVX-512 is absent on this CPU and is not selected. GEMM N=2048: Nexus 330, 362 and 369 GFLOPS; TensorFlow 389 GFLOPS. Deleted experiments (an Nc tile that repacks A, MASM 6×16, hybrid GEMM at N=512) are not the current kernel. cuDNN is force-disabled. Conv shape 8×3×32×32, 16 filters, 3×3, pad 1: CPU 6.39, device-resident kernel 48.7, host round-trip 19.0 GFLOPS. That conv figure is not compared with PyTorch.',
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
                  {t('GEMM N=2048 (ölçüldü)', 'GEMM N=2048 (measured)')}
                </span>
                <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary">
                  Ryzen 7 5800H · no AVX-512
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
                <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  {t(
                    'Conv 8×3×32×32, 16 filtre, 3×3, dolgu 1',
                    'Conv 8×3×32×32, 16 filters, 3×3, pad 1',
                  )}
                </p>
                {convData.map((item, idx) => (
                  <div key={idx} className="space-y-1.5">
                    <div className="flex items-center justify-between font-mono text-xs">
                      <span className="max-w-[280px] truncate text-foreground">{item.name}</span>
                      <span className="font-bold text-primary">{item.time}</span>
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
                <CodeBlock code={benchCode} filename="build.ps1" />
                <a
                  href="/docs/performans"
                  className="inline-flex items-center gap-2 font-mono text-xs font-bold text-primary hover:underline"
                >
                  {t(
                    '21 Ekim CPU tablosu ve 1 Ekim GPU protokolü',
                    '21 October CPU table and the 1 October GPU protocol',
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

'use client'

import { useEffect, useState } from 'react'
import { ArrowRight, CheckCircle, ShieldAlert, Zap, Cpu } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Reveal } from '@/components/reveal'
import { useLanguage } from '@/lib/language-context'

interface Stage {
  key: string
  call: string
  title: { tr: string; en: string }
  desc: { tr: string; en: string }
  note: { tr: string; en: string }
  status: string
  grid: string[]
}

const STAGES: Stage[] = [
  {
    key: 'alloc',
    call: 'Matrix(M, N, MemoryMode::device_only)',
    title: { tr: '01. Cihaz İçi Tahsis', en: '01. Device-Only Allocation' },
    desc: {
      tr: 'Bellek yalnızca GPU üzerinde ayrılır; RAM tarafında ayna tampon oluşturulmaz.',
      en: 'Allocates exclusively in GPU VRAM; host RAM mirror is skipped entirely.',
    },
    note: { tr: 'Sıfır PCIe Trafiği', en: 'Zero PCIe Overhead' },
    status: 'DEVICE_ONLY',
    grid: ['0.00', '0.00', '0.00', '0.00'],
  },
  {
    key: 'gemm',
    call: 'left * right  (cuBLAS GEMM)',
    title: { tr: '02. cuBLAS Matris Çarpımı', en: '02. cuBLAS GEMM Engine' },
    desc: {
      tr: 'NVIDIA cuBLAS ve TF32 Tensor Core ile donanım sınırında paralel matris çarpımı.',
      en: 'Parallel matrix product saturating hardware capacity via cuBLAS and TF32 Tensor Cores.',
    },
    note: { tr: 'TF32 Tensor Cores', en: 'TF32 Tensor Cores' },
    status: 'COMPUTE',
    grid: ['1.85', '-0.4', '3.12', '2.40'],
  },
  {
    key: 'fused',
    call: 'fused_bias_gelu(prod, bias)',
    title: { tr: '03. Fused Kernel Zinciri', en: '03. Fused Kernel Chain' },
    desc: {
      tr: 'Bias ekleme ve GeLU aktivasyonu tek bir CUDA kernelında birleştirilir.',
      en: 'Bias addition and GeLU non-linearity collapsed into a single CUDA kernel pass.',
    },
    note: { tr: '1 Memory Pass', en: '1 Memory Pass' },
    status: 'FUSED_ACTIVE',
    grid: ['1.79', '0.00', '3.08', '2.37'],
  },
  {
    key: 'async',
    call: 'argmax_async(out, slot, cb)',
    title: { tr: '04. Asenkron İndirgeme', en: '04. Async Stream Reduction' },
    desc: {
      tr: 'CPU ana iş parçacığı bloklanmadan sonuç pinned host belleğine yazılır.',
      en: 'Reduces directly into pinned host memory without blocking the host CPU thread.',
    },
    note: { tr: 'Stream Pool Slot #2', en: 'Stream Pool Slot #2' },
    status: 'NON_BLOCKING',
    grid: ['idx=2', 'max=3.08', 'async', 'ready'],
  },
  {
    key: 'download',
    call: 'out.download()  [Fail-Fast]',
    title: { tr: '05. Senkronizasyon & Okuma', en: '05. Safe Host Materialization' },
    desc: {
      tr: 'Host tarafında at() çağırmadan önce download() zorunludur; bayat veri okunamaz.',
      en: 'download() is verified before at() access; eliminates silent stale-read bugs.',
    },
    note: { tr: 'Fail-Safe Verified', en: 'Fail-Safe Verified' },
    status: 'SYNCHRONIZED',
    grid: ['1.79', '0.00', '3.08', '2.37'],
  },
]

export function PipelineDemo() {
  const [step, setStep] = useState(0)
  const { lang, t } = useLanguage()

  useEffect(() => {
    const id = setInterval(() => setStep((s) => (s + 1) % STAGES.length), 2800)
    return () => clearInterval(id)
  }, [])

  const current = STAGES[step]

  return (
    <section className="relative border-y border-border/80 bg-gradient-to-b from-card/30 via-card/10 to-card/30 py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Reveal className="mx-auto max-w-3xl text-center">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
            <Zap className="size-3.5" />
            <span>{t('HIZLI & GÜVENLİ YAŞAM DÖNGÜSÜ', 'FAST & SAFE EXECUTION PIPELINE')}</span>
          </div>
          <h2 className="mt-4 text-balance text-3xl font-extrabold tracking-tight sm:text-4xl">
            {t('GPU Belleğinde Uçtan Uca Kesintisiz Akış', 'Zero-Stall GPU-Resident Execution Pipeline')}
          </h2>
          <p className="mt-4 text-pretty text-base text-muted-foreground sm:text-lg">
            {t(
              'Tüm matrisler VRAM üzerinde yaşar. cuBLAS ve fused kernel’lar zincirlenir, gereksiz PCIe gecikmeleri ortadan kalkar.',
              'Buffers remain strictly resident in VRAM. cuBLAS and fused chains run without unnecessary PCIe transfer bottlenecks.',
            )}
          </p>
        </Reveal>

        {/* Pipeline Step Cards */}
        <Reveal className="mt-14" delay={80}>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {STAGES.map((stage, i) => {
              const isActive = i === step
              return (
                <button
                  key={stage.key}
                  type="button"
                  onClick={() => setStep(i)}
                  className={cn(
                    'group flex flex-col justify-between rounded-2xl border p-4 text-left transition-all duration-300',
                    isActive
                      ? 'border-primary bg-primary/[0.08] shadow-[0_0_25px_rgba(110,231,183,0.2)] scale-[1.02]'
                      : 'border-border/70 bg-card/50 opacity-70 hover:opacity-100 hover:border-border',
                  )}
                >
                  <div>
                    <div className="flex items-center justify-between border-b border-border/50 pb-2.5">
                      <span
                        className={cn(
                          'font-mono text-xs font-bold',
                          isActive ? 'text-primary' : 'text-muted-foreground',
                        )}
                      >
                        {stage.title[lang]}
                      </span>
                      <span className="rounded bg-secondary/80 px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">
                        {stage.note[lang]}
                      </span>
                    </div>

                    <code className="mt-3 block font-mono text-xs font-medium text-foreground truncate">
                      {stage.call}
                    </code>

                    {/* Matrix Grid Visualization */}
                    <div className="mt-3 grid grid-cols-2 gap-1">
                      {stage.grid.map((val, idx) => (
                        <div
                          key={idx}
                          className={cn(
                            'flex h-7 items-center justify-center rounded font-mono text-[11px] font-semibold tabular-nums transition-colors',
                            isActive
                              ? 'bg-primary/20 text-primary border border-primary/30'
                              : 'bg-black/30 text-muted-foreground border border-border/30',
                          )}
                        >
                          {val}
                        </div>
                      ))}
                    </div>

                    <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
                      {stage.desc[lang]}
                    </p>
                  </div>

                  <div className="mt-3 pt-2 border-t border-border/40 flex items-center justify-between font-mono text-[10px]">
                    <span className={isActive ? 'text-primary' : 'text-muted-foreground'}>
                      {stage.status}
                    </span>
                    {isActive ? (
                      <CheckCircle className="size-3 text-primary animate-pulse" />
                    ) : (
                      <span className="text-muted-foreground/50">0{i + 1}</span>
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        </Reveal>
      </div>
    </section>
  )
}

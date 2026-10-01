'use client'

import { useEffect, useMemo, useState } from 'react'
import { DocPageRenderer } from '@/components/docs/doc-page-renderer'
import { useDocContent } from '@/lib/hooks/use-doc-content'
import { useLanguage } from '@/lib/hooks/use-language'
import type { DocPageData } from '@/lib/types/doc'

type Cell = { us: number; gflops: number }
type EngineId = 'tuned' | 'plan' | 'cublas' | 'cublaslt'
type Precision = 'fp32' | 'tf32'

type FairFile = {
  timestamp: string
  device: { name: string }
  shapes: { id: string; label: string }[]
  gpu: Record<Precision, Record<EngineId, Record<string, Cell>>>
  numpy_wall_ms: Record<string, number>
  gpu_fp32_plan_wall_ms: Record<string, number>
}

const ENGINES: { id: EngineId; tr: string; en: string; bar: string }[] = [
  { id: 'tuned', tr: 'MatrixFlash-Pro tuned plan', en: 'MatrixFlash-Pro tuned plan', bar: 'from-emerald-500 to-primary' },
  { id: 'plan', tr: 'MatrixFlash-Pro plan', en: 'MatrixFlash-Pro plan', bar: 'from-teal-500 to-emerald-400' },
  { id: 'cublas', tr: 'Ham cuBLAS', en: 'Raw cuBLAS', bar: 'from-cyan-500 to-cyan-300' },
  { id: 'cublaslt', tr: 'Ham cuBLASLt', en: 'Raw cuBLASLt', bar: 'from-violet-500 to-violet-300' },
]

const QUICK = ['512', '1024', '2048', '1x1024x4096', '64x1024x512']

export default function PerformansPage() {
  const { data, loading, error } = useDocContent<DocPageData>('/api/docs/benchmarks-comparison')
  return (
    <>
      <FairChart />
      <DocPageRenderer data={data} loading={loading} error={error} />
    </>
  )
}

function FairChart() {
  const { t, lang } = useLanguage()
  const tr = lang === 'tr'
  const [precision, setPrecision] = useState<Precision>('fp32')
  const [shape, setShape] = useState('1024')
  const [file, setFile] = useState<FairFile | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    fetch('/data/benchmarks/fair_gemm.json')
      .then((r) => {
        if (!r.ok) throw new Error('fair_gemm HTTP ' + r.status)
        return r.json() as Promise<FairFile>
      })
      .then((json) => {
        if (alive) setFile(json)
      })
      .catch((e: unknown) => {
        if (alive) setLoadError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      alive = false
    }
  }, [])

  const rows = useMemo(() => {
    if (!file) return []
    const bucket = file.gpu[precision]
    return ENGINES.map((engine) => {
      const cell = bucket[engine.id][shape]
      return { ...engine, us: cell?.us ?? 0, gflops: cell?.gflops ?? 0 }
    }).sort((a, b) => b.gflops - a.gflops)
  }, [file, precision, shape])

  const maxGflops = rows.reduce((m, r) => Math.max(m, r.gflops), 0.001)
  const gpuWall = file?.gpu_fp32_plan_wall_ms[shape]
  const cpuWall = file?.numpy_wall_ms[shape]
  const shapes = file?.shapes ?? []
  const quick = shapes.filter((s) => QUICK.includes(s.id))
  const rest = shapes.filter((s) => !QUICK.includes(s.id))

  return (
    <div className="mb-8 space-y-6">
      <section className="rounded-2xl border border-border/80 bg-card/50 p-6 shadow-2xl backdrop-blur-xl">
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border/60 pb-3">
          <div>
            <span className="font-mono text-xs font-bold text-foreground">
              {t('Aynı hassasiyette GEMM — 1 Ekim 2026', 'Same-precision GEMM — 1 October 2026')}
            </span>
            <p className="mt-1 max-w-xl font-mono text-[10px] leading-relaxed text-muted-foreground">
              {t(
                'Önceden ayrılmış tampon, CUDA event medyanı, 10 ısınma / 30 tekrar × 2 tur. PyTorch ve CuPy kurulu değildi.',
                'Preallocated buffers, CUDA-event median, 10 warmups / 30 repeats × 2 runs. PyTorch and CuPy were not installed.',
              )}
            </p>
          </div>
          <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary">
            {file?.device.name ?? 'RTX 3070 Laptop GPU'}
            {file?.timestamp ? ` · ${file.timestamp}` : ''}
          </span>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {(['fp32', 'tf32'] as Precision[]).map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPrecision(p)}
              className={
                p === precision
                  ? 'rounded-lg border border-primary/50 bg-primary/15 px-3 py-1 font-mono text-xs font-bold text-primary'
                  : 'rounded-lg border border-transparent px-3 py-1 font-mono text-xs text-muted-foreground hover:bg-white/[0.04] hover:text-foreground'
              }
            >
              {p === 'fp32' ? 'FP32 pedantic' : 'TF32 Tensor Core'}
            </button>
          ))}
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {[...quick, ...rest].map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setShape(s.id)}
              className={
                s.id === shape
                  ? 'rounded-lg border border-primary/50 bg-primary/15 px-3 py-1 font-mono text-xs font-bold text-primary'
                  : 'rounded-lg border border-transparent px-3 py-1 font-mono text-xs text-muted-foreground hover:bg-white/[0.04] hover:text-foreground'
              }
            >
              {s.label}
            </button>
          ))}
        </div>

        <div className="mt-5 space-y-4">
          {loadError && (
            <p className="font-mono text-xs text-destructive">
              {t('Ölçü JSON yüklenemedi', 'Failed to load JSON')}: {loadError}
            </p>
          )}
          {rows.map((r) => (
            <div key={r.id} className="space-y-1.5">
              <div className="flex items-center justify-between gap-3 font-mono text-xs">
                <span className="truncate text-foreground">{tr ? r.tr : r.en}</span>
                <span className="shrink-0 font-bold text-primary">
                  {(r.us / 1000).toFixed(3)} ms · {r.gflops.toFixed(0)} GFLOPS
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary/80">
                <div
                  className={'h-full rounded-full bg-gradient-to-r ' + r.bar}
                  style={{ width: Math.max(4, (r.gflops / maxGflops) * 100) + '%' }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      {gpuWall != null && cpuWall != null && (
        <section className="rounded-2xl border border-border/80 bg-card/50 p-6 shadow-2xl backdrop-blur-xl">
          <div className="border-b border-border/60 pb-3">
            <span className="font-mono text-xs font-bold text-foreground">
              {t('CPU duvar saati bağlamı', 'CPU wall-clock context')}
            </span>
            <p className="mt-1 font-mono text-[10px] text-muted-foreground">
              {t(
                'Strict FP32 plan duvar saati ile NumPy 2.3.5 @ OpenBLAS. Transfer yok. 64³ altında CPU daha hızlıdır.',
                'Strict FP32 plan wall time versus NumPy 2.3.5 @ OpenBLAS. No transfer. Below 64³ the CPU is faster.',
              )}
            </p>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <WallCard label="MatrixFlash-Pro FP32 plan" ms={gpuWall} emphasize={gpuWall <= cpuWall} />
            <WallCard label="NumPy @ OpenBLAS" ms={cpuWall} emphasize={cpuWall < gpuWall} />
          </div>
        </section>
      )}
    </div>
  )
}

function WallCard({ label, ms, emphasize }: { label: string; ms: number; emphasize?: boolean }) {
  return (
    <div className="rounded-xl border border-border/70 bg-background/40 p-4">
      <div className="font-mono text-[11px] text-muted-foreground">{label}</div>
      <div className={emphasize ? 'mt-1 font-mono text-lg font-bold text-primary' : 'mt-1 font-mono text-lg font-bold text-foreground'}>
        {ms.toFixed(3)} ms
      </div>
    </div>
  )
}

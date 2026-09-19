'use client'

import { useEffect, useState } from 'react'
import { DocPageRenderer } from '@/components/docs/doc-page-renderer'
import { useDocContent } from '@/lib/hooks/use-doc-content'
import { useLanguage } from '@/lib/hooks/use-language'
import type { DocPageData } from '@/lib/types/doc'

type BenchRow = {
  bench: string
  case: string
  size: number
  ms_median: number
  throughput: number
  unit: string
  note: string
}

type BenchFile = {
  library: string
  device: { name: string } | null
  results: BenchRow[]
}

const GEMM_SIZES = [256, 512, 1024, 2048]

const BAR_STYLES: Record<string, string> = {
  mflash: 'from-emerald-500 to-primary',
  raw: 'from-cyan-500 to-cyan-300',
  naive: 'from-amber-500 to-amber-300',
  cpu: 'from-zinc-500 to-zinc-400',
}

function shortName(note: string): string {
  if (note.startsWith('mflash')) return 'MatrixFlash-Pro'
  if (note.startsWith('raw cublas')) return 'ham cuBLAS'
  if (note.startsWith('naive global')) return 'naive GPU'
  if (note.startsWith('cpu single')) return 'CPU tek-thread'
  return note
}

function barKey(note: string): string {
  if (note.startsWith('mflash')) return 'mflash'
  if (note.startsWith('raw cublas')) return 'raw'
  if (note.startsWith('naive global')) return 'naive'
  return 'cpu'
}

function BenchmarkComparison() {
  const { t } = useLanguage()
  const [size, setSize] = useState<number>(1024)
  const [data, setData] = useState<BenchFile | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    fetch('/data/benchmarks/comparison.json')
      .then((r) => {
        if (!r.ok) throw new Error('HTTP ' + r.status)
        return r.json() as Promise<BenchFile>
      })
      .then((j) => {
        if (alive) setData(j)
      })
      .catch((e: unknown) => {
        if (alive) setLoadError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      alive = false
    }
  }, [])

  const rows = (data?.results ?? []).filter(
    (r) => r.size === size && r.unit === 'GFLOPS' && !r.case.startsWith('mlp'),
  )
  const maxGflops = rows.reduce((m, r) => Math.max(m, r.throughput), 1)

  return (
    <section className="rounded-2xl border border-border/80 bg-card/50 p-6 shadow-2xl backdrop-blur-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 pb-3">
        <span className="font-mono text-xs font-bold text-foreground">
          {t('GEMM Karsilastirma - olculdu', 'GEMM Comparison - measured')}
        </span>
        <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary">
          {data?.device?.name ?? 'RTX 3070 Laptop GPU'}
        </span>
      </div>
      <div className="mt-4 flex flex-wrap gap-1.5">
        {GEMM_SIZES.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSize(s)}
            className={
              s === size
                ? 'rounded-lg border border-primary/50 bg-primary/15 px-3 py-1 font-mono text-xs font-bold text-primary'
                : 'rounded-lg border border-transparent px-3 py-1 font-mono text-xs text-muted-foreground hover:bg-white/[0.04] hover:text-foreground'
            }
          >
            {s}x{s}
          </button>
        ))}
      </div>
      <div className="mt-5 space-y-4">
        {loadError && (
          <p className="font-mono text-xs text-destructive">
            {t('Olcu JSON yuklenemedi', 'Failed to load JSON')}: {loadError}
          </p>
        )}
        {!data && !loadError && (
          <p className="font-mono text-xs text-muted-foreground">
            {t('Veri yukleniyor...', 'Loading...')}
          </p>
        )}
        {rows.map((r, i) => (
          <div key={i} className="space-y-1.5">
            <div className="flex items-center justify-between font-mono text-xs">
              <span className="truncate text-foreground">{shortName(r.note)}</span>
              <span className="font-bold text-primary">
                {r.ms_median.toFixed(3)} ms
              </span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary/80">
              <div
                className={'h-full rounded-full bg-gradient-to-r ' + BAR_STYLES[barKey(r.note)]}
                style={{ width: Math.max(4, (r.throughput / maxGflops) * 100) + '%' }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

/** Full /docs/performans page: JSON doc content + live chart above it. */
export default function PerformansPage() {
  const { data, loading, error } = useDocContent<DocPageData>('/api/docs/benchmarks-comparison')
  return (
    <>
      <div className="mb-6">
        <BenchmarkComparison />
      </div>
      <DocPageRenderer data={data} loading={loading} error={error} />
    </>
  )
}


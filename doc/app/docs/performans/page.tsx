'use client'

import { useEffect, useMemo, useState } from 'react'
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
  timestamp?: string
  device: { name: string } | null
  results: BenchRow[]
}

type RivalEngine = {
  id: string
  name: string
  kind: string
  measured: boolean
  by_size?: Record<string, { ms_median: number; gflops: number; note?: string }>
  band?: string
  source?: string
}

type RivalsFile = {
  engines: RivalEngine[]
  sizes: number[]
}

const GEMM_SIZES = [256, 512, 1024, 2048]

const BAR_STYLES: Record<string, string> = {
  into: 'from-emerald-500 to-primary',
  mflash: 'from-teal-500 to-emerald-400',
  raw: 'from-cyan-500 to-cyan-300',
  naive: 'from-amber-500 to-amber-300',
  cpu: 'from-zinc-500 to-zinc-400',
}

const RIVAL_COLORS = [
  'from-emerald-500 to-primary',
  'from-cyan-500 to-cyan-300',
  'from-violet-500 to-violet-300',
  'from-amber-500 to-amber-300',
  'from-rose-500 to-rose-300',
  'from-sky-500 to-sky-300',
]

function shortName(note: string, tr: boolean): string {
  if (note.includes('multiply_into')) return 'MatrixFlash-Pro multiply_into'
  if (note.startsWith('mflash')) return 'MatrixFlash-Pro operator*'
  if (note.startsWith('raw cublas')) return tr ? 'ham cuBLAS' : 'raw cuBLAS'
  if (note.startsWith('naive global')) return tr ? 'naive GPU' : 'naive GPU'
  if (note.startsWith('cpu single')) return tr ? 'CPU tek-thread' : 'CPU single-thread'
  return note
}

function barKey(note: string): string {
  if (note.includes('multiply_into')) return 'into'
  if (note.startsWith('mflash')) return 'mflash'
  if (note.startsWith('raw cublas')) return 'raw'
  if (note.startsWith('naive global')) return 'naive'
  return 'cpu'
}

function sortRank(note: string): number {
  const order = ['cpu', 'naive', 'mflash', 'into', 'raw']
  return order.indexOf(barKey(note))
}

function BenchmarkComparison() {
  const { t, lang } = useLanguage()
  const tr = lang === 'tr'
  const [size, setSize] = useState<number>(1024)
  const [data, setData] = useState<BenchFile | null>(null)
  const [rivals, setRivals] = useState<RivalsFile | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    Promise.all([
      fetch('/data/benchmarks/comparison.json').then((r) => {
        if (!r.ok) throw new Error('comparison HTTP ' + r.status)
        return r.json() as Promise<BenchFile>
      }),
      fetch('/data/benchmarks/rivals.json')
        .then((r) => (r.ok ? (r.json() as Promise<RivalsFile>) : null))
        .catch(() => null),
    ])
      .then(([cmp, riv]) => {
        if (!alive) return
        setData(cmp)
        setRivals(riv)
      })
      .catch((e: unknown) => {
        if (alive) setLoadError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      alive = false
    }
  }, [])

  const rows = useMemo(() => {
    const filtered = (data?.results ?? []).filter(
      (r) => r.size === size && r.unit === 'GFLOPS' && !r.case.startsWith('mlp'),
    )
    return [...filtered].sort((a, b) => sortRank(a.note) - sortRank(b.note))
  }, [data, size])

  const maxGflops = rows.reduce((m, r) => Math.max(m, r.throughput), 1)
  const into = rows.find((r) => r.note.includes('multiply_into'))
  const raw = rows.find((r) => r.note.startsWith('raw cublas'))
  const ratio =
    into && raw && raw.throughput > 0
      ? ((into.throughput / raw.throughput) * 100).toFixed(0)
      : null

  const measuredRivals = useMemo(() => {
    if (!rivals) return []
    return rivals.engines
      .filter((e) => e.measured && e.by_size?.[String(size)])
      .map((e) => {
        const cell = e.by_size![String(size)]
        return { name: e.name, kind: e.kind, gflops: cell.gflops, ms: cell.ms_median }
      })
      .sort((a, b) => b.gflops - a.gflops)
  }, [rivals, size])

  const maxRival = measuredRivals.reduce((m, r) => Math.max(m, r.gflops), 1)

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-border/80 bg-card/50 p-6 shadow-2xl backdrop-blur-xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 pb-3">
          <div>
            <span className="font-mono text-xs font-bold text-foreground">
              {t('GEMM vs cuBLAS — ölçüldü', 'GEMM vs cuBLAS — measured')}
            </span>
            {ratio && (
              <p className="mt-1 font-mono text-[10px] text-primary">
                {t(
                  `multiply_into ≈ cuBLAS’ın %${ratio}’i${Number(ratio) >= 100 ? ' (geçiyor)' : ''}`,
                  `multiply_into ≈ ${ratio}% of cuBLAS${Number(ratio) >= 100 ? ' (ahead)' : ''}`,
                )}
              </p>
            )}
          </div>
          <div className="flex flex-col items-end gap-1">
            <span className="rounded bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary">
              {data?.device?.name ?? 'RTX 3070 Laptop GPU'}
            </span>
            {data?.timestamp && (
              <span className="font-mono text-[10px] text-muted-foreground">{data.timestamp}</span>
            )}
          </div>
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
              {s}×{s}
            </button>
          ))}
        </div>
        <div className="mt-5 space-y-4">
          {loadError && (
            <p className="font-mono text-xs text-destructive">
              {t('Ölçü JSON yüklenemedi', 'Failed to load JSON')}: {loadError}
            </p>
          )}
          {rows.map((r, i) => (
            <div key={i} className="space-y-1.5">
              <div className="flex items-center justify-between font-mono text-xs">
                <span className="truncate text-foreground">{shortName(r.note, tr)}</span>
                <span className="shrink-0 font-bold text-primary">
                  {r.ms_median.toFixed(3)} ms · {r.throughput.toFixed(0)} GFLOPS
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

      {measuredRivals.length > 0 && (
        <section className="rounded-2xl border border-border/80 bg-card/50 p-6 shadow-2xl backdrop-blur-xl">
          <div className="border-b border-border/60 pb-3">
            <span className="font-mono text-xs font-bold text-foreground">
              {t(
                `Piyasa motorları — ${size}×${size} (ölçülenler)`,
                `Market engines — ${size}×${size} (measured)`,
              )}
            </span>
            <p className="mt-1 font-mono text-[10px] text-muted-foreground">
              {t(
                'MatrixFlash-Pro, cuBLAS, cublasLt, NumPy@OpenBLAS — aynı makine',
                'MatrixFlash-Pro, cuBLAS, cublasLt, NumPy@OpenBLAS — same machine',
              )}
            </p>
          </div>
          <div className="mt-5 space-y-4">
            {measuredRivals.map((r, i) => (
              <div key={r.name} className="space-y-1.5">
                <div className="flex items-center justify-between font-mono text-xs">
                  <span className="truncate text-foreground">
                    {r.name}
                    <span className="ml-2 text-muted-foreground">({r.kind})</span>
                  </span>
                  <span className="shrink-0 font-bold text-primary">
                    {r.ms.toFixed(3)} ms · {r.gflops.toFixed(0)} GFLOPS
                  </span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-secondary/80">
                  <div
                    className={
                      'h-full rounded-full bg-gradient-to-r ' + RIVAL_COLORS[i % RIVAL_COLORS.length]
                    }
                    style={{ width: Math.max(4, (r.gflops / maxRival) * 100) + '%' }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

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

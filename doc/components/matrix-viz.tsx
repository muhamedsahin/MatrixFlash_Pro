'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

type MatrixVizProps = {
  rows?: number
  cols?: number
  className?: string
  label?: string
  /** highlight moving cell */
  animate?: boolean
}

/**
 * Decorative animated matrix: a grid of live-updating float cells with a
 * moving highlight that sweeps through, evoking GPU parallel compute.
 */
export function MatrixViz({
  rows = 4,
  cols = 4,
  className,
  label,
  animate = true,
}: MatrixVizProps) {
  const total = rows * cols
  const [values, setValues] = useState<number[]>(() =>
    Array.from({ length: total }, () => 0),
  )
  const [active, setActive] = useState(-1)

  useEffect(() => {
    // init once on client to avoid hydration mismatch
    setValues(Array.from({ length: total }, () => Math.random() * 2 - 1))
  }, [total])

  useEffect(() => {
    if (!animate) return
    let i = 0
    const id = setInterval(() => {
      i = (i + 1) % total
      setActive(i)
      setValues((prev) => {
        const next = [...prev]
        next[i] = Math.random() * 2 - 1
        return next
      })
    }, 220)
    return () => clearInterval(id)
  }, [animate, total])

  return (
    <div className={cn('inline-flex flex-col gap-2', className)}>
      {label && (
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {label}
        </span>
      )}
      <div className="relative rounded-lg border border-border bg-card/60 p-2">
        <div
          className="grid gap-1.5"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        >
          {values.map((v, i) => {
            const isActive = i === active
            return (
              <div
                key={i}
                className={cn(
                  'grid h-9 w-11 place-items-center rounded-md font-mono text-[11px] tabular-nums transition-all duration-300',
                  isActive
                    ? 'bg-primary/20 text-primary ring-1 ring-primary/50'
                    : 'bg-secondary/40 text-muted-foreground',
                )}
                style={{
                  opacity: 0.55 + Math.min(Math.abs(v), 1) * 0.45,
                }}
              >
                {v.toFixed(2)}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

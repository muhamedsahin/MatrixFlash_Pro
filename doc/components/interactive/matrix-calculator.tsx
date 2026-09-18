'use client'

import React, { useState, useEffect } from 'react'
import { Play, RotateCcw, SkipForward, Cpu, CheckCircle2, Sparkles, Layers } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLanguage } from '@/lib/language-context'

export function MatrixCalculator() {
  const { lang, t } = useLanguage()

  // Preset matrices
  // A is 2x3, B is 3x2 => C is 2x2
  const matA = [
    [1, 2, 3],
    [4, 5, 6],
  ]
  const matB = [
    [7, 8],
    [9, 1],
    [2, 4],
  ]

  const rowsA = matA.length
  const colsA = matA[0].length
  const colsB = matB[0].length
  const totalSteps = rowsA * colsB

  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const activeRow = Math.floor(currentStep / colsB)
  const activeCol = currentStep % colsB

  // Auto-play effect
  useEffect(() => {
    let timer: NodeJS.Timeout
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, 1600)
    }
    return () => clearInterval(timer)
  }, [isPlaying, totalSteps])

  // Compute cell calculation breakdown
  const terms: { a: number; b: number; prod: number }[] = []
  let cellSum = 0
  for (let k = 0; k < colsA; k++) {
    const valA = matA[activeRow][k]
    const valB = matB[k][activeCol]
    const prod = valA * valB
    cellSum += prod
    terms.push({ a: valA, b: valB, prod })
  }

  // Precompute whole C matrix up to currentStep
  const matC: (number | null)[][] = Array.from({ length: rowsA }, () =>
    Array.from({ length: colsB }, () => null),
  )

  for (let step = 0; step <= currentStep; step++) {
    const r = Math.floor(step / colsB)
    const c = step % colsB
    let sum = 0
    for (let k = 0; k < colsA; k++) {
      sum += matA[r][k] * matB[k][c]
    }
    matC[r][c] = sum
  }

  const handleNext = () => {
    if (currentStep < totalSteps - 1) {
      setCurrentStep((s) => s + 1)
    }
  }

  const handleReset = () => {
    setIsPlaying(false)
    setCurrentStep(0)
  }

  return (
    <div className="relative overflow-hidden rounded-2xl border border-primary/30 bg-gradient-to-b from-card/90 to-card/40 p-6 shadow-2xl backdrop-blur-xl">
      {/* Glow header badge */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border/70 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary/15 text-primary shadow-[0_0_15px_rgba(110,231,183,0.3)]">
            <Cpu className="size-5" />
          </div>
          <div>
            <h3 className="font-mono text-base font-bold text-foreground flex items-center gap-2">
              <span>{t('İnteraktif Matris Çarpımı Simülatörü', 'Interactive Matmul Simulator')}</span>
              <span className="rounded-full bg-primary/10 border border-primary/30 px-2 py-0.5 text-[10px] font-semibold text-primary">
                C = A × B
              </span>
            </h3>
            <p className="text-xs text-muted-foreground">
              {t(
                'Her bir sonuç hücresinin satır-sütun nokta çarpımıyla nasıl hesaplandığını adım adım izleyin.',
                'Watch step-by-step how each output cell is evaluated via row-column dot product.',
              )}
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center gap-1.5 rounded-lg border border-primary/40 bg-primary/15 px-3 py-1.5 font-mono text-xs font-semibold text-primary transition-all hover:bg-primary/25"
          >
            <Play className={cn('size-3.5', isPlaying && 'fill-current')} />
            {isPlaying ? t('Durdur', 'Pause') : t('Otomatik Oynat', 'Auto Play')}
          </button>
          <button
            type="button"
            onClick={handleNext}
            disabled={currentStep >= totalSteps - 1}
            className="flex items-center gap-1.5 rounded-lg border border-border bg-secondary/50 px-3 py-1.5 font-mono text-xs text-foreground transition-all hover:bg-secondary disabled:opacity-40"
          >
            <SkipForward className="size-3.5" />
            {t('Sonraki Adım', 'Next Step')}
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border border-border bg-secondary/50 p-2 text-muted-foreground transition-all hover:bg-secondary hover:text-foreground"
            title={t('Sıfırla', 'Reset')}
          >
            <RotateCcw className="size-3.5" />
          </button>
        </div>
      </div>

      {/* Main visualization grid */}
      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3 lg:items-center">
        {/* Matrix A */}
        <div className="flex flex-col items-center">
          <div className="mb-2 font-mono text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
            <span className="text-primary font-bold">A</span> (2×3)
            <span className="text-[10px] text-primary/70">
              [{t('Aktif Satır', 'Active Row')}: {activeRow + 1}]
            </span>
          </div>
          <div className="relative rounded-xl border border-border/80 bg-[oklch(0.12_0.008_160)] p-3 shadow-inner">
            <div className="grid grid-cols-3 gap-2">
              {matA.map((row, r) =>
                row.map((val, c) => {
                  const isHighlighted = r === activeRow
                  return (
                    <div
                      key={`a-${r}-${c}`}
                      className={cn(
                        'flex size-12 items-center justify-center rounded-lg font-mono text-sm font-bold transition-all duration-300',
                        isHighlighted
                          ? 'border-2 border-primary bg-primary/20 text-primary shadow-[0_0_12px_rgba(110,231,183,0.4)] scale-105'
                          : 'border border-border/40 bg-card/60 text-muted-foreground',
                      )}
                    >
                      {val}
                    </div>
                  )
                }),
              )}
            </div>
          </div>
        </div>

        {/* Multiplier symbol & Matrix B */}
        <div className="flex flex-col items-center">
          <div className="mb-2 font-mono text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
            <span className="text-accent font-bold">B</span> (3×2)
            <span className="text-[10px] text-accent/70">
              [{t('Aktif Sütun', 'Active Col')}: {activeCol + 1}]
            </span>
          </div>
          <div className="relative rounded-xl border border-border/80 bg-[oklch(0.12_0.008_160)] p-3 shadow-inner">
            <div className="grid grid-cols-2 gap-2">
              {matB.map((row, r) =>
                row.map((val, c) => {
                  const isHighlighted = c === activeCol
                  return (
                    <div
                      key={`b-${r}-${c}`}
                      className={cn(
                        'flex size-12 items-center justify-center rounded-lg font-mono text-sm font-bold transition-all duration-300',
                        isHighlighted
                          ? 'border-2 border-accent bg-accent/20 text-accent shadow-[0_0_12px_rgba(56,189,248,0.4)] scale-105'
                          : 'border border-border/40 bg-card/60 text-muted-foreground',
                      )}
                    >
                      {val}
                    </div>
                  )
                }),
              )}
            </div>
          </div>
        </div>

        {/* Equals & Matrix C */}
        <div className="flex flex-col items-center">
          <div className="mb-2 font-mono text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
            <span className="text-foreground font-bold">C</span> (2×2)
            <span className="text-[10px] text-emerald-400 font-semibold">
              [C({activeRow + 1},{activeCol + 1})]
            </span>
          </div>
          <div className="relative rounded-xl border border-border/80 bg-[oklch(0.12_0.008_160)] p-3 shadow-inner">
            <div className="grid grid-cols-2 gap-2">
              {matC.map((row, r) =>
                row.map((val, c) => {
                  const isCurrent = r === activeRow && c === activeCol
                  const isEvaluated = val !== null
                  return (
                    <div
                      key={`c-${r}-${c}`}
                      className={cn(
                        'flex size-12 items-center justify-center rounded-lg font-mono text-sm font-bold transition-all duration-300',
                        isCurrent
                          ? 'border-2 border-emerald-400 bg-emerald-500/25 text-emerald-300 shadow-[0_0_20px_rgba(52,211,153,0.6)] animate-pulse scale-110'
                          : isEvaluated
                            ? 'border border-primary/40 bg-primary/10 text-foreground'
                            : 'border border-dashed border-border/40 bg-card/30 text-muted-foreground/30',
                      )}
                    >
                      {val !== null ? val : '?'}
                    </div>
                  )
                }),
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Formula & Dot Product Breakdown */}
      <div className="mt-6 rounded-xl border border-border/80 bg-[oklch(0.11_0.008_160)] p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="font-mono text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {t('Nokta Çarpım Açılımı', 'Dot Product Expansion')} · C[{activeRow}][{activeCol}]
          </span>
          <span className="font-mono text-xs text-primary">
            {t('Adım', 'Step')} {currentStep + 1} / {totalSteps}
          </span>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 font-mono text-xs sm:text-sm">
          <span className="text-muted-foreground">C({activeRow + 1},{activeCol + 1}) = </span>
          {terms.map((item, idx) => (
            <React.Fragment key={idx}>
              <span className="rounded bg-primary/15 px-1.5 py-0.5 text-primary">
                {item.a}
              </span>
              <span className="text-muted-foreground">×</span>
              <span className="rounded bg-accent/15 px-1.5 py-0.5 text-accent">
                {item.b}
              </span>
              {idx < terms.length - 1 && <span className="text-muted-foreground">+</span>}
            </React.Fragment>
          ))}
          <span className="text-muted-foreground">=</span>
          <span className="font-bold text-emerald-400">
            {terms.map((t) => t.prod).join(' + ')} = {cellSum}
          </span>
        </div>

        {/* GPU parallel note */}
        <div className="mt-3 flex items-center gap-2 border-t border-border/50 pt-2 text-[11px] text-muted-foreground">
          <Layers className="size-3.5 text-primary shrink-0" />
          <span>
            {t(
              'GPU üzerinde cuBLAS ve shared-memory tiling sayesinde bu hücrelerin tümü paralel iş parçacıkları (threads) tarafından AYNI ANDA hesaplanır!',
              'On GPU with cuBLAS and shared-memory tiling, all these output cells are computed concurrently by parallel threads at the SAME TIME!',
            )}
          </span>
        </div>
      </div>
    </div>
  )
}


'use client'

import { useEffect, useState } from 'react'
import { Logo } from '@/components/logo'

const BOOT_LINES = [
  '[ 0.001 ] matrix_pro :: creating CUDA context ...',
  '[ 0.042 ] cuBLAS handle ready · TF32 tensor math enabled',
  '[ 0.089 ] device RNG (Philox) seeded · 0 host round-trips',
  '[ 0.133 ] compiling fused kernel chains ... 100%',
  '[ 0.187 ] autograd tape engine online',
  '[ 0.201 ] welcome to MatrixFlash-Pro v2.0.0-PRO',
]

/**
 * One-shot terminal-boot intro animation. Plays once per browser session
 * (sessionStorage guard), then fades the whole page in.
 */
export function IntroOverlay() {
  const [visibleLines, setVisibleLines] = useState(0)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState<'boot' | 'fading' | 'done'>('boot')

  useEffect(() => {
    let seen = false
    try {
      seen = sessionStorage.getItem('mfp_intro_done') === '1'
    } catch {
      // storage unavailable — still play once
    }
    if (seen) {
      setPhase('done')
      return
    }

    document.documentElement.style.overflow = 'hidden'

    const lineTimer = setInterval(() => {
      setVisibleLines((v) => {
        if (v >= BOOT_LINES.length) {
          clearInterval(lineTimer)
          return v
        }
        return v + 1
      })
    }, 260)

    const progressTimer = setInterval(() => {
      setProgress((p) => Math.min(p + Math.random() * 14 + 6, 100))
    }, 150)

    const finishTimer = setTimeout(() => {
      setProgress(100)
      setPhase('fading')
      try {
        sessionStorage.setItem('mfp_intro_done', '1')
      } catch {
        // ignore
      }
      document.documentElement.style.overflow = ''
    }, 260 * BOOT_LINES.length + 500)

    const doneTimer = setTimeout(() => setPhase('done'), 260 * BOOT_LINES.length + 500 + 650)

    return () => {
      clearInterval(lineTimer)
      clearInterval(progressTimer)
      clearTimeout(finishTimer)
      clearTimeout(doneTimer)
      document.documentElement.style.overflow = ''
    }
  }, [])

  if (phase === 'done') return null

  return (
    <div
      className={`fixed inset-0 z-[100] flex items-center justify-center bg-background transition-opacity duration-700 ${
        phase === 'fading' ? 'opacity-0' : 'opacity-100'
      }`}
    >
      {/* Background grid */}
      <div className="absolute inset-0 grid-bg radial-fade opacity-60" aria-hidden />
      <div className="absolute left-1/2 top-1/2 h-[420px] w-[680px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/10 blur-[130px]" aria-hidden />

      <div className="relative w-[min(92vw,560px)]">
        {/* Logo pulse */}
        <div className="mb-8 flex flex-col items-center gap-4">
          <div className="relative animate-pulse-glow">
            <Logo className="size-16" />
          </div>
          <p className="font-mono text-xs uppercase tracking-[0.35em] text-primary text-glow">
            MatrixFlash-Pro
          </p>
        </div>

        {/* Terminal boot log */}
        <div className="rounded-xl border border-border/80 bg-[oklch(0.13_0.008_160)]/90 p-4 font-mono text-[11px] leading-relaxed text-muted-foreground shadow-2xl backdrop-blur">
          {BOOT_LINES.slice(0, visibleLines).map((line, i) => (
            <p key={i} className="animate-fade-up">
              <span className="text-primary">{line.slice(0, 9)}</span>
              {line.slice(9)}
            </p>
          ))}
          <p className="mt-1 flex items-center gap-1 text-primary">
            <span className="inline-block h-3 w-2 animate-caret-blink bg-primary" />
          </p>

          {/* Progress */}
          <div className="mt-4 flex items-center gap-3">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary/60">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-[width] duration-150"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="w-10 text-right text-[10px] tabular-nums text-primary">
              {Math.round(progress)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

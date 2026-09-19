'use client'

import { useState } from 'react'
import dynamic from 'next/dynamic'
import { Sparkles, Activity, Layers, Cpu } from 'lucide-react'
import { cn } from '@/lib/utils'

const Hero3DCanvas = dynamic(
  () => import('@/components/three/hero-scene').then((m) => m.Hero3DCanvas),
  {
    ssr: false,
    loading: () => (
      <div className="h-full w-full animate-pulse rounded-3xl border border-primary/20 bg-primary/[0.03] flex items-center justify-center">
        <div className="flex items-center gap-2 font-mono text-xs text-primary/60">
          <Activity className="size-4 animate-spin" />
          <span>3D HYPERTENSOR ENGINE INITIALIZING...</span>
        </div>
      </div>
    ),
  },
)

/** Client-only WebGL canvas wrapper with rich cyber HUD overlays. */
export function Hero3D() {
  const [hudActive, setHudActive] = useState(true)

  return (
    <div className="group relative h-[400px] w-full sm:h-[480px] lg:h-[580px] rounded-3xl border border-border/70 bg-black/40 p-2 shadow-2xl backdrop-blur-sm overflow-hidden surface-shine">
      {/* Corner HUD accent brackets */}
      <div className="pointer-events-none absolute left-3 top-3 size-4 border-l-2 border-t-2 border-primary/80 z-20" />
      <div className="pointer-events-none absolute right-3 top-3 size-4 border-r-2 border-t-2 border-primary/80 z-20" />
      <div className="pointer-events-none absolute left-3 bottom-3 size-4 border-l-2 border-b-2 border-primary/80 z-20" />
      <div className="pointer-events-none absolute right-3 bottom-3 size-4 border-r-2 border-b-2 border-primary/80 z-20" />

      {/* 3D WebGL Canvas */}
      <Hero3DCanvas />

      {/* Top HUD bar */}
      <div className="pointer-events-none absolute left-5 top-5 right-5 flex items-center justify-between z-20">
        <div className="flex items-center gap-2 rounded-xl border border-primary/30 bg-background/70 px-3 py-1.5 font-mono text-[11px] text-primary backdrop-blur-md shadow-[0_0_15px_rgba(74,222,128,0.15)]">
          <span className="size-2 rounded-full bg-primary animate-ping" />
          <span className="font-semibold">64-CELL TENSOR HYPERCORE</span>
          <span className="text-muted-foreground">· N-D STRIDED</span>
        </div>

        <div className="hidden sm:flex items-center gap-1.5 rounded-xl border border-accent/30 bg-background/70 px-3 py-1.5 font-mono text-[11px] text-accent backdrop-blur-md shadow-[0_0_15px_rgba(34,211,238,0.15)]">
          <Cpu className="size-3.5" />
          <span>TF32 TENSOR CORE PATH</span>
        </div>
      </div>

      {/* Bottom HUD bar with mouse & scroll hint */}
      <div className="pointer-events-none absolute left-5 bottom-5 right-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 z-20">
        <div className="flex items-center gap-2 rounded-xl border border-border/80 bg-background/70 px-3 py-1.5 font-mono text-[10px] text-muted-foreground backdrop-blur-md">
          <Layers className="size-3 text-primary" />
          <span>SCROLL & MOUSE REACTIVE 3D MATRIX</span>
        </div>

        <div className="flex items-center gap-2 rounded-xl border border-border/80 bg-background/70 px-3 py-1.5 font-mono text-[10px] text-primary backdrop-blur-md">
          <Sparkles className="size-3 text-lime-400" />
          <span>CUBLAS-CLASS · ZERO LAUNCH OVERHEAD</span>
        </div>
      </div>
    </div>
  )
}

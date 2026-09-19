'use client'

import dynamic from 'next/dynamic'

const Hero3DCanvas = dynamic(
  () => import('@/components/three/hero-scene').then((m) => m.Hero3DCanvas),
  {
    ssr: false,
    loading: () => (
      <div className="h-full w-full animate-pulse rounded-3xl border border-border/60 bg-primary/[0.04]" />
    ),
  },
)

/** Client-only WebGL canvas wrapper (avoids SSR with three.js). */
export function Hero3D() {
  return (
    <div className="relative h-[380px] w-full sm:h-[460px] lg:h-[560px]">
      <Hero3DCanvas />
      {/* HUD captions floating over the 3D scene */}
      <div className="pointer-events-none absolute left-4 top-4 rounded-lg border border-border/60 bg-background/50 px-3 py-1.5 font-mono text-[10px] text-primary backdrop-blur">
        GPU LATTICE · 64 CELLS
      </div>
      <div className="pointer-events-none absolute bottom-4 right-4 rounded-lg border border-border/60 bg-background/50 px-3 py-1.5 font-mono text-[10px] text-accent backdrop-blur">
        MOUSE REACTIVE · WEBGL
      </div>
    </div>
  )
}

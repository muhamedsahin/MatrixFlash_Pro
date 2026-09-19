'use client'

import { useCallback, useEffect, useRef } from 'react'

/**
 * Pointer-reactive 3D tilt: rotateX/rotateY CSS transform toward the pointer,
 * with spring-like smoothing. Attach the returned handlers to any element.
 */
export function useTilt(maxTiltDeg = 8, smoothing = 0.12) {
  const ref = useRef<HTMLElement | null>(null)
  const target = useRef({ rx: 0, ry: 0 })
  const current = useRef({ rx: 0, ry: 0 })
  const rafId = useRef(0)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const tick = () => {
      current.current.rx += (target.current.rx - current.current.rx) * smoothing
      current.current.ry += (target.current.ry - current.current.ry) * smoothing
      el.style.transform = `perspective(900px) rotateX(${current.current.rx.toFixed(3)}deg) rotateY(${current.current.ry.toFixed(3)}deg)`
      rafId.current = requestAnimationFrame(tick)
    }
    rafId.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId.current)
  }, [])

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const el = e.currentTarget as HTMLElement
      const rect = el.getBoundingClientRect()
      const px = (e.clientX - rect.left) / rect.width - 0.5
      const py = (e.clientY - rect.top) / rect.height - 0.5
      target.current.ry = px * maxTiltDeg * 2
      target.current.rx = -py * maxTiltDeg * 2
    },
    [maxTiltDeg],
  )

  const onPointerLeave = useCallback(() => {
    target.current.rx = 0
    target.current.ry = 0
  }, [])

  return { ref, onPointerMove, onPointerLeave }
}

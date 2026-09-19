'use client'

import { useEffect, useRef, useState } from 'react'

export type MousePosition = { x: number; y: number; nx: number; ny: number }

/**
 * Tracks the pointer in pixels (x, y) and normalized device coords
 * (nx, ny ∈ [-1, 1], centered). Used to make 3D visuals react to the mouse.
 */
export function useMousePosition(): MousePosition {
  const [pos, setPos] = useState<MousePosition>({ x: 0, y: 0, nx: 0, ny: 0 })
  const raf = useRef(0)

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      cancelAnimationFrame(raf.current)
      raf.current = requestAnimationFrame(() => {
        const nx = (e.clientX / window.innerWidth) * 2 - 1
        const ny = (e.clientY / window.innerHeight) * 2 - 1
        setPos({ x: e.clientX, y: e.clientY, nx, ny })
      })
    }

    window.addEventListener('pointermove', onMove, { passive: true })
    return () => {
      window.removeEventListener('pointermove', onMove)
      cancelAnimationFrame(raf.current)
    }
  }, [])

  return pos
}

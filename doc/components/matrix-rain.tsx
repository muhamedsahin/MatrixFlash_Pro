'use client'

import { useEffect, useRef } from 'react'

type MatrixRainProps = {
  className?: string
  /** 0-1, opacity of the whole canvas */
  opacity?: number
  fontSize?: number
}

/**
 * Canvas-based "digital rain" of falling numbers/glyphs, tinted with the
 * NVIDIA-green primary. Purely decorative background.
 */
export function MatrixRain({
  className,
  opacity = 0.5,
  fontSize = 16,
}: MatrixRainProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const glyphs =
      '01λΣΠ∑∏∫√×÷±≈≠∞ABCDEF0123456789{}[]<>=+-*/.'.split('')

    let width = 0
    let height = 0
    let columns = 0
    let drops: number[] = []
    let dpr = 1

    const setup = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      const parent = canvas.parentElement
      width = parent?.clientWidth ?? window.innerWidth
      height = parent?.clientHeight ?? window.innerHeight
      canvas.width = width * dpr
      canvas.height = height * dpr
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      columns = Math.floor(width / fontSize)
      drops = Array.from({ length: columns }, () =>
        Math.floor((Math.random() * height) / fontSize),
      )
    }

    setup()

    let raf = 0
    let last = 0
    const step = (time: number) => {
      raf = requestAnimationFrame(step)
      if (time - last < 60) return
      last = time

      ctx.fillStyle = 'rgba(10, 15, 10, 0.22)'
      ctx.fillRect(0, 0, width, height)
      ctx.font = `${fontSize}px var(--font-jetbrains-mono, monospace)`

      for (let i = 0; i < drops.length; i++) {
        const char = glyphs[Math.floor(Math.random() * glyphs.length)]
        const x = i * fontSize
        const y = drops[i] * fontSize

        // Leading glyph glows brighter
        const bright = Math.random() > 0.975
        ctx.fillStyle = bright
          ? 'rgba(210, 255, 190, 0.95)'
          : 'rgba(140, 220, 90, 0.75)'
        ctx.fillText(char, x, y)

        if (y > height && Math.random() > 0.975) {
          drops[i] = 0
        }
        drops[i]++
      }
    }
    raf = requestAnimationFrame(step)

    const onResize = () => setup()
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
    }
  }, [fontSize])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={className}
      style={{ opacity }}
    />
  )
}

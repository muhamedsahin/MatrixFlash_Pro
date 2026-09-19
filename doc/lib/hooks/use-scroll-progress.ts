'use client'

import { useEffect, useState } from 'react'

/** Reading progress through the page, 0 → 1. Drives the top progress bar. */
export function useScrollProgress(): number {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let raf = 0
    const update = () => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        const scrollTop = window.scrollY
        const height = document.documentElement.scrollHeight - window.innerHeight
        setProgress(height > 0 ? Math.min(scrollTop / height, 1) : 0)
      })
    }

    update()
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    return () => {
      window.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
      cancelAnimationFrame(raf)
    }
  }, [])

  return progress
}

'use client'

import { useScrollProgress } from '@/lib/hooks/use-scroll-progress'

/** Fixed reading-progress bar pinned under the site nav. */
export function ScrollProgressBar() {
  const progress = useScrollProgress()

  return (
    <div
      className="fixed inset-x-0 top-0 z-[60] h-[3px] bg-transparent"
      role="progressbar"
      aria-valuenow={Math.round(progress * 100)}
    >
      <div
        className="h-full origin-left bg-gradient-to-r from-primary via-accent to-primary shadow-[0_0_12px_rgba(74,222,128,0.7)] transition-[width] duration-100 ease-linear"
        style={{ width: `${progress * 100}%` }}
      />
    </div>
  )
}

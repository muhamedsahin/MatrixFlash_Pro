'use client'

import { useState } from 'react'
import { PanelLeft, X } from 'lucide-react'
import { DocsSidebar } from './docs-sidebar'
import { useLanguage } from '@/lib/language-context'

export function DocsMobileNav() {
  const [open, setOpen] = useState(false)
  const { lang, t } = useLanguage()

  return (
    <div className="lg:hidden">
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 rounded-lg border border-border/80 bg-card/70 px-3.5 py-1.5 font-mono text-xs text-foreground shadow-sm backdrop-blur-md hover:border-primary/40"
      >
        <PanelLeft className="size-4 text-primary" />
        <span>{t('İçindekiler', 'Table of Contents')}</span>
      </button>

      {open && (
        <div className="fixed inset-0 z-[60] flex">
          <button
            type="button"
            aria-label="Kapat"
            className="fixed inset-0 bg-background/80 backdrop-blur-md"
            onClick={() => setOpen(false)}
          />
          <div className="relative z-10 flex h-full w-80 max-w-[85vw] flex-col border-r border-border bg-background p-5 shadow-2xl animate-fade-up">
            <div className="mb-6 flex items-center justify-between border-b border-border/60 pb-3">
              <span className="font-mono text-sm font-bold text-foreground">
                MatrixFlash<span className="text-primary">-Pro</span>
              </span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Kapat"
                className="rounded-lg border border-border p-1 text-muted-foreground hover:bg-card hover:text-foreground"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto pr-2">
              <DocsSidebar onNavigate={() => setOpen(false)} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

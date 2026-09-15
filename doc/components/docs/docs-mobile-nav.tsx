'use client'

import { useState } from 'react'
import { PanelLeft, X } from 'lucide-react'
import { DocsSidebar } from './docs-sidebar'

export function DocsMobileNav() {
  const [open, setOpen] = useState(false)

  return (
    <div className="lg:hidden">
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 rounded-md border border-border bg-secondary/50 px-3 py-1.5 font-mono text-xs text-muted-foreground"
      >
        <PanelLeft className="size-4" />
        Menü
      </button>

      {open && (
        <div className="fixed inset-0 z-[60]">
          <button
            type="button"
            aria-label="Kapat"
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <div className="absolute left-0 top-0 h-full w-72 overflow-y-auto border-r border-border bg-background p-5">
            <div className="mb-6 flex items-center justify-between">
              <span className="font-mono text-sm font-bold">
                Doküman<span className="text-primary">tasyon</span>
              </span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Kapat"
                className="rounded-md p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="size-5" />
              </button>
            </div>
            <DocsSidebar onNavigate={() => setOpen(false)} />
          </div>
        </div>
      )}
    </div>
  )
}

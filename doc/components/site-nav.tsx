'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { Menu, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { site, topNav } from '@/lib/site'
import { Logo } from '@/components/logo'
import { GithubIcon } from '@/components/github-icon'

export function SiteNav() {
  const pathname = usePathname()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => setOpen(false), [pathname])

  return (
    <header
      className={cn(
        'fixed inset-x-0 top-0 z-50 transition-all duration-300',
        scrolled
          ? 'border-b border-border bg-background/80 backdrop-blur-xl'
          : 'border-b border-transparent',
      )}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo className="size-8" />
          <span className="font-mono text-sm font-bold tracking-tight">
            MatrixFlash<span className="text-primary">-Pro</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {topNav.map((item) => {
            const active =
              item.href === '/'
                ? pathname === '/'
                : pathname.startsWith(item.href)
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'rounded-md px-3 py-2 text-sm transition-colors',
                  active
                    ? 'text-foreground'
                    : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="flex items-center gap-2">
          <a
            href={site.github}
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-2 rounded-md border border-border bg-secondary/50 px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground sm:flex"
          >
            <GithubIcon className="size-4" />
            <span className="font-mono text-xs">GitHub</span>
          </a>
          <button
            type="button"
            className="rounded-md p-2 text-muted-foreground hover:text-foreground md:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-label="Menü"
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </div>

      {open && (
        <div className="border-t border-border bg-background/95 backdrop-blur-xl md:hidden">
          <nav className="flex flex-col p-4">
            {topNav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="rounded-md px-3 py-2.5 text-sm text-muted-foreground hover:bg-secondary hover:text-foreground"
              >
                {item.label}
              </Link>
            ))}
            <a
              href={site.github}
              target="_blank"
              rel="noreferrer"
              className="mt-2 flex items-center gap-2 rounded-md border border-border px-3 py-2.5 text-sm"
            >
              <GithubIcon className="size-4" /> GitHub
            </a>
          </nav>
        </div>
      )}
    </header>
  )
}

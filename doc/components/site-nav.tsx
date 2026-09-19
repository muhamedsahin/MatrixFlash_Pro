'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import { Menu, X, Globe, Sparkles, FileText } from 'lucide-react'
import { cn } from '@/lib/utils'
import { site, topNav } from '@/lib/site'
import { Logo } from '@/components/logo'
import { GithubIcon } from '@/components/github-icon'
import { useLanguage } from '@/lib/language-context'

export function SiteNav() {
  const pathname = usePathname()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const { lang, setLang } = useLanguage()

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
          ? 'border-b border-border/60 bg-background/75 shadow-[0_8px_40px_oklch(0_0_0_/_0.45)] backdrop-blur-2xl'
          : 'border-b border-transparent bg-background/25 backdrop-blur-md',
      )}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand Logo */}
        <Link href="/" className="group flex items-center gap-2.5">
          <div className="relative">
            <Logo className="size-8 transition-transform duration-300 group-hover:scale-110" />
            <div className="absolute inset-0 -z-10 rounded-full bg-primary/30 blur-md opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
          </div>
          <div className="flex flex-col">
            <span className="font-mono text-sm font-bold tracking-tight text-foreground">
              MatrixFlash<span className="text-primary text-glow">-Pro</span>
            </span>
            <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground/80">
              v{site.version} · CUDA
            </span>
          </div>
        </Link>

        {/* Desktop Navigation */}
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
                  'relative rounded-lg px-3.5 py-1.5 text-sm font-medium transition-all duration-200',
                  active
                    ? 'bg-primary/10 text-primary glow-primary'
                    : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground',
                )}
              >
                {item.label[lang]}
              </Link>
            )
          })}
        </nav>

        {/* Action controls (Language Switcher + GitHub + Mobile menu toggle) */}
        <div className="flex items-center gap-2.5">
          {/* Language Toggle Pill */}
          <div className="flex items-center rounded-full border border-border/80 bg-card/70 p-0.5 backdrop-blur-md">
            <button
              type="button"
              onClick={() => setLang('tr')}
              className={cn(
                'rounded-full px-2.5 py-1 text-[11px] font-mono font-semibold transition-all',
                lang === 'tr'
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
              )}
              title="Türkçe"
            >
              TR
            </button>
            <button
              type="button"
              onClick={() => setLang('en')}
              className={cn(
                'rounded-full px-2.5 py-1 text-[11px] font-mono font-semibold transition-all',
                lang === 'en'
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
              )}
              title="English"
            >
              EN
            </button>
          </div>

          {/* PDF Technical Manual Download */}
          <a
            href="/MatrixFlash_Pro_Technical_Manual.pdf"
            download="MatrixFlash_Pro_Technical_Manual.pdf"
            className="hidden items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs font-mono font-medium text-emerald-400 hover:bg-emerald-500/20 hover:border-emerald-500/60 transition-all sm:flex"
            title="Download MatrixFlash-Pro Technical Manual (57 Pages PDF)"
          >
            <FileText className="size-3.5 text-emerald-400" />
            <span>PDF (57s)</span>
          </a>

          {/* GitHub link */}
          <a
            href={site.github}
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-2 rounded-lg border border-border bg-secondary/40 px-3.5 py-1.5 text-sm font-medium text-foreground transition-all duration-200 hover:border-primary/40 hover:bg-secondary/70 hover:shadow-[0_0_15px_rgba(110,231,183,0.15)] sm:flex"
          >
            <GithubIcon className="size-4 text-primary" />
            <span className="font-mono text-xs">GitHub</span>
          </a>

          {/* Mobile menu toggle */}
          <button
            type="button"
            className="rounded-lg border border-border bg-card/60 p-2 text-muted-foreground transition-colors hover:text-foreground md:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-label="Menü"
          >
            {open ? <X className="size-5 text-primary" /> : <Menu className="size-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {open && (
        <div className="border-t border-border/80 bg-background/95 p-4 backdrop-blur-2xl md:hidden animate-fade-up">
          <div className="flex items-center justify-between pb-3 border-b border-border/60">
            <span className="text-xs font-mono text-muted-foreground">
              {lang === 'tr' ? 'DİL SEÇİMİ' : 'LANGUAGE'}
            </span>
            <div className="flex items-center rounded-full border border-border bg-card p-0.5">
              <button
                type="button"
                onClick={() => setLang('tr')}
                className={cn(
                  'rounded-full px-3 py-1 text-xs font-mono font-semibold',
                  lang === 'tr'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground',
                )}
              >
                Türkçe (TR)
              </button>
              <button
                type="button"
                onClick={() => setLang('en')}
                className={cn(
                  'rounded-full px-3 py-1 text-xs font-mono font-semibold',
                  lang === 'en'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground',
                )}
              >
                English (EN)
              </button>
            </div>
          </div>

          <nav className="flex flex-col gap-1 pt-3">
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
                    'flex items-center justify-between rounded-lg px-3.5 py-2.5 text-sm font-medium transition-colors',
                    active
                      ? 'bg-primary/10 text-primary font-semibold'
                      : 'text-muted-foreground hover:bg-card hover:text-foreground',
                  )}
                >
                  <span>{item.label[lang]}</span>
                  {active && <span className="size-1.5 rounded-full bg-primary" />}
                </Link>
              )
            })}

            <a
              href="/MatrixFlash_Pro_Technical_Manual.pdf"
              download="MatrixFlash_Pro_Technical_Manual.pdf"
              className="mt-3 flex items-center justify-center gap-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-4 py-2.5 text-sm font-medium text-emerald-400 hover:bg-emerald-500/20 transition-colors"
            >
              <FileText className="size-4 text-emerald-400" />
              <span>{lang === 'tr' ? 'Teknik Manuel PDF (57 Sayfa)' : 'Technical Manual PDF (57 Pages)'}</span>
            </a>

            <a
              href={site.github}
              target="_blank"
              rel="noreferrer"
              className="mt-2 flex items-center justify-center gap-2 rounded-lg border border-border bg-card/60 px-4 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-card"
            >
              <GithubIcon className="size-4 text-primary" /> GitHub Repository
            </a>
          </nav>
        </div>
      )}
    </header>
  )
}

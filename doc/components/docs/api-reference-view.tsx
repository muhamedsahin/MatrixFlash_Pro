'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { Search, ChevronLeft, ChevronRight, LayoutGrid, X, ArrowUp, Layers } from 'lucide-react'
import { ApiEntry } from '@/components/docs/api-entry'
import { DocSection, PageHeader, Callout } from '@/components/docs/doc-ui'
import { T } from '@/components/docs/doc-blocks'
import { Reveal } from '@/components/reveal'
import type { DocPageData } from '@/lib/types/doc'
import { useLanguage } from '@/lib/hooks/use-language'
import { cn } from '@/lib/utils'

export function ApiReferenceView({ data }: { data: DocPageData | null }) {
  const { lang, t } = useLanguage()
  const [searchTerm, setSearchTerm] = useState('')
  const [activeCategory, setActiveCategory] = useState<string>('all')
  const [isScrolled, setIsScrolled] = useState(false)
  const [showAllCategories, setShowAllCategories] = useState(false)
  const categoryScrollRef = useRef<HTMLDivElement>(null)

  const sections = data?.sections ?? []

  const categories = useMemo(
    () => [
      { id: 'all', label: { tr: 'Tümü', en: 'All' } },
      ...sections.map((s) => ({ id: s.id, label: s.title })),
    ],
    [sections],
  )

  // Track window scroll for shrinking the sticky filter bar
  useEffect(() => {
    let ticking = false
    const handleScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          setIsScrolled(window.scrollY > 160)
          ticking = false
        })
        ticking = true
      }
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Smooth scroll helper with header offset
  const scrollToSection = (id: string, updateUrl = true) => {
    if (id === 'all') {
      window.scrollTo({ top: 0, behavior: 'smooth' })
      setActiveCategory('all')
      if (updateUrl) window.history.replaceState(null, '', window.location.pathname)
      return
    }

    const el = document.getElementById(id)
    if (el) {
      const yOffset = -85
      const y = el.getBoundingClientRect().top + window.pageYOffset + yOffset
      window.scrollTo({ top: y, behavior: 'smooth' })
      if (updateUrl) window.history.replaceState(null, '', `#${id}`)
      setActiveCategory(id)

      el.classList.remove('section-highlight')
      void el.offsetWidth
      el.classList.add('section-highlight')
    }
  }

  // Handle hash from URL or custom sidebar navigation
  useEffect(() => {
    if (!data) return

    const handleTargetHash = (hash: string) => {
      if (!hash) return
      const cleanHash = hash.replace('#', '')
      setActiveCategory(cleanHash)
      // Clear search so the section is definitely visible
      if (searchTerm) setSearchTerm('')
      setTimeout(() => {
        scrollToSection(cleanHash, false)
      }, 120)
    }

    // Initial load check
    if (window.location.hash) {
      handleTargetHash(window.location.hash)
    }

    const onHashChange = () => handleTargetHash(window.location.hash)
    const onCustomNav = (e: Event) => {
      const customEvent = e as CustomEvent<{ hash: string }>
      if (customEvent.detail?.hash) {
        handleTargetHash(customEvent.detail.hash)
      }
    }

    window.addEventListener('hashchange', onHashChange)
    window.addEventListener('doc-navigate-hash', onCustomNav)
    return () => {
      window.removeEventListener('hashchange', onHashChange)
      window.removeEventListener('doc-navigate-hash', onCustomNav)
    }
  }, [data])

  // Scroll category strip horizontally
  const scrollCategoryTrack = (direction: 'left' | 'right') => {
    if (!categoryScrollRef.current) return
    const offset = direction === 'left' ? -220 : 220
    categoryScrollRef.current.scrollBy({ left: offset, behavior: 'smooth' })
  }

  // Filter entries based on search term
  const filtered = useMemo(() => {
    const term = searchTerm.trim().toLowerCase()
    return sections
      .map((s) => ({
        ...s,
        entries: (s.entries ?? []).filter((e) => {
          if (!term) return true
          return (
            e.name.toLowerCase().includes(term) ||
            e.signature.toLowerCase().includes(term) ||
            (e.badge ?? '').toLowerCase().includes(term) ||
            e.text[lang].toLowerCase().includes(term)
          )
        }),
      }))
      .filter((s) => s.entries.length > 0)
  }, [sections, searchTerm, lang])

  if (!data) return null

  const activeCategoryTitle =
    categories.find((c) => c.id === activeCategory)?.label[lang] ?? t('Tümü', 'All')

  return (
    <article className="space-y-6">
      <PageHeader
        eyebrow={data.meta.eyebrow[lang]}
        title={data.meta.title[lang]}
        description={data.meta.description[lang]}
      />

      <Callout type="info" title={t('Ad Alanı ve Başlıklar', 'Namespace & Header Notice')}>
        {t(
          'Tüm fonksiyonlar ve sınıflar matrix_pro ad alanı altındadır. Tek şemsiye başlık `#include \"matrix_pro/matrix_pro.hpp\"` ile tüm modüllere erişebilirsiniz.',
          'All types and routines reside under the matrix_pro namespace. Include the single umbrella header `#include \"matrix_pro/matrix_pro.hpp\"` to access the full API.',
        )}
      </Callout>

      {/* 
        Sleek, Dynamic, Shrinking Sticky Filter Bar 
        - When at top: Full elegant search + 1-row scrollable category track (~84px height)
        - When scrolled down: Compresses into ultra-compact HUD bar (~52px height)
        - NEVER blocks the screen!
      */}
      <div
        className={cn(
          'sticky top-16 z-30 transition-all duration-300',
          isScrolled ? 'top-14 sm:top-16 pt-1' : '',
        )}
      >
        <div
          className={cn(
            'rounded-2xl border border-border/80 bg-background/85 shadow-xl backdrop-blur-2xl transition-all duration-300',
            isScrolled ? 'py-2 px-3 shadow-[0_8px_32px_rgba(0,0,0,0.5)] border-primary/25 bg-background/90' : 'p-3 sm:p-4',
          )}
        >
          {/* Main search row */}
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-primary/70" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder={t(
                  'Fonksiyon veya sınıf ara (örn: matmul, relu_, svd, Variable)...',
                  'Search function or class (e.g. matmul, relu_, svd, Variable)...',
                )}
                className={cn(
                  'w-full rounded-xl border border-border/80 bg-black/50 pl-9 pr-8 font-mono text-sm text-foreground placeholder:text-muted-foreground/50 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition-all',
                  isScrolled ? 'py-1.5 text-xs' : 'py-2 text-sm',
                )}
              />
              {searchTerm && (
                <button
                  type="button"
                  onClick={() => setSearchTerm('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-full p-0.5 text-muted-foreground hover:bg-white/10 hover:text-foreground"
                >
                  <X className="size-3.5" />
                </button>
              )}
            </div>

            {/* Active category chip on scrolled view */}
            {isScrolled && activeCategory !== 'all' && (
              <button
                type="button"
                onClick={() => scrollToSection(activeCategory)}
                className="hidden sm:inline-flex items-center gap-1.5 rounded-lg border border-primary/40 bg-primary/15 px-2.5 py-1 font-mono text-xs text-primary shadow-[0_0_12px_rgba(74,222,128,0.2)] shrink-0"
              >
                <Layers className="size-3" />
                <span className="max-w-[120px] truncate">{activeCategoryTitle}</span>
              </button>
            )}

            {/* Toggle grid view for categories */}
            <button
              type="button"
              onClick={() => setShowAllCategories((v) => !v)}
              title={t('Tüm kategorileri göster/gizle', 'Toggle category grid')}
              className={cn(
                'rounded-xl border border-border/80 bg-black/40 p-2 text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary',
                showAllCategories && 'border-primary/60 bg-primary/10 text-primary',
                isScrolled ? 'p-1.5' : 'p-2',
              )}
            >
              <LayoutGrid className="size-4" />
            </button>

            {/* Quick scroll to top if scrolled */}
            {isScrolled && (
              <button
                type="button"
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                title={t('En başa dön', 'Back to top')}
                className="rounded-xl border border-border/80 bg-black/40 p-1.5 text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
              >
                <ArrowUp className="size-4" />
              </button>
            )}
          </div>

          {/* Search match stats */}
          {searchTerm && (
            <div className="mt-2 flex items-center justify-between font-mono text-xs text-primary">
              <span>
                {filtered.reduce((acc, s) => acc + s.entries.length, 0)} {t('sonuç bulundu', 'matches found')}
              </span>
              <button
                type="button"
                onClick={() => setSearchTerm('')}
                className="text-xs text-muted-foreground hover:text-foreground underline"
              >
                {t('Temizle', 'Clear')}
              </button>
            </div>
          )}

          {/* 
            Single-line horizontal category pill bar with scroll chevrons
            Only takes ~34px height! Never wraps into a 300px block!
          */}
          {!showAllCategories && (
            <div className={cn('relative mt-2 flex items-center', isScrolled && 'hidden sm:flex')}>
              <button
                type="button"
                onClick={() => scrollCategoryTrack('left')}
                className="absolute left-0 z-10 hidden rounded-md bg-background/90 p-1 text-muted-foreground hover:text-primary sm:flex"
                aria-label="Scroll left"
              >
                <ChevronLeft className="size-4" />
              </button>

              <div
                ref={categoryScrollRef}
                className="flex flex-nowrap items-center gap-1.5 overflow-x-auto no-scrollbar scroll-smooth py-0.5 px-0.5 sm:px-6 w-full"
                style={{
                  maskImage:
                    'linear-gradient(to right, transparent, black 16px, black calc(100% - 16px), transparent)',
                }}
              >
                {categories.map((cat) => (
                  <button
                    key={cat.id}
                    type="button"
                    onClick={() => scrollToSection(cat.id)}
                    className={cn(
                      'shrink-0 rounded-lg px-2.5 py-1 font-mono text-xs font-medium transition-all whitespace-nowrap',
                      activeCategory === cat.id
                        ? 'border border-primary/50 bg-primary/15 text-primary shadow-[0_0_12px_rgba(74,222,128,0.25)]'
                        : 'border border-transparent text-muted-foreground hover:bg-white/[0.04] hover:text-foreground',
                    )}
                  >
                    {cat.label[lang]}
                  </button>
                ))}
              </div>

              <button
                type="button"
                onClick={() => scrollCategoryTrack('right')}
                className="absolute right-0 z-10 hidden rounded-md bg-background/90 p-1 text-muted-foreground hover:text-primary sm:flex"
                aria-label="Scroll right"
              >
                <ChevronRight className="size-4" />
              </button>
            </div>
          )}

          {/* Expanded category grid (Only visible if user clicks the grid icon) */}
          {showAllCategories && (
            <div className="mt-3 max-h-56 overflow-y-auto rounded-xl border border-border/60 bg-black/60 p-3 no-scrollbar">
              <div className="flex flex-wrap gap-1.5">
                {categories.map((cat) => (
                  <button
                    key={cat.id}
                    type="button"
                    onClick={() => {
                      scrollToSection(cat.id)
                      setShowAllCategories(false)
                    }}
                    className={cn(
                      'rounded-lg px-2.5 py-1 font-mono text-xs font-medium transition-all',
                      activeCategory === cat.id
                        ? 'border border-primary/50 bg-primary/20 text-primary shadow-[0_0_12px_rgba(74,222,128,0.3)]'
                        : 'border border-border/40 text-muted-foreground hover:border-primary/40 hover:text-foreground',
                    )}
                  >
                    {cat.label[lang]}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Render all sections */}
      {filtered.map((section, idx) => (
        <Reveal key={section.id} delay={idx * 25}>
          <DocSection id={section.id} title={section.title[lang]}>
            {section.intro && <T value={section.intro} />}
          </DocSection>

          <div className="space-y-4">
            {section.entries.map((entry) => (
              <ApiEntry
                key={entry.name}
                name={entry.name}
                signature={entry.signature}
                badge={entry.badge}
                returns={entry.returns}
                params={entry.params?.map((p) => ({
                  name: p.name,
                  type: p.type,
                  desc: p.desc[lang],
                }))}
                example={entry.example}
              >
                <T value={entry.text} />
              </ApiEntry>
            ))}
          </div>
        </Reveal>
      ))}

      {filtered.length === 0 && (
        <div className="rounded-2xl border border-border/80 bg-card/40 p-10 text-center font-mono text-sm text-muted-foreground">
          {t('Aramanızla eşleşen API öğesi bulunamadı.', 'No API entries matched your search.')}
        </div>
      )}
    </article>
  )
}

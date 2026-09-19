'use client'

import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { PageHeader, DocSection } from '@/components/docs/doc-ui'
import { BlockList, T } from '@/components/docs/doc-blocks'
import { ApiEntry } from '@/components/docs/api-entry'
import { Reveal } from '@/components/reveal'
import type { DocPageData } from '@/lib/types/doc'
import { useLanguage } from '@/lib/hooks/use-language'
import { useEffect } from 'react'

function DocSkeleton() {
  return (
    <div className="animate-pulse space-y-6" aria-busy="true">
      <div className="h-3 w-40 rounded bg-secondary/70" />
      <div className="h-10 w-3/4 rounded bg-secondary/70" />
      <div className="h-4 w-2/3 rounded bg-secondary/50" />
      <div className="space-y-3 pt-6">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-4 rounded bg-secondary/40" style={{ width: `${95 - i * 8}%` }} />
        ))}
      </div>
      <div className="h-40 rounded-xl border border-border bg-card/40" />
    </div>
  )
}

export function DocPageRenderer({
  data,
  loading,
  error,
  next,
}: {
  data: DocPageData | null
  loading: boolean
  error: Error | null
  next?: { href: string; label: { tr: string; en: string } }
}) {
  const { lang, t } = useLanguage()

  // ✅ TÜM hook'lar burada, herhangi bir return'den ÖNCE.
  useEffect(() => {
    if (!data) return
    const scrollToTarget = (hash: string) => {
      if (!hash) return
      const cleanHash = hash.replace('#', '')
      const el = document.getElementById(cleanHash)
      if (el) {
        const yOffset = -85
        const y = el.getBoundingClientRect().top + window.pageYOffset + yOffset
        window.scrollTo({ top: y, behavior: 'smooth' })
        el.classList.remove('section-highlight')
        void el.offsetWidth
        el.classList.add('section-highlight')
      }
    }

    if (window.location.hash) {
      setTimeout(() => scrollToTarget(window.location.hash), 120)
    }

    const onHashChange = () => scrollToTarget(window.location.hash)
    const onCustomNav = (e: Event) => {
      const customEvent = e as CustomEvent<{ hash: string }>
      if (customEvent.detail?.hash) scrollToTarget(customEvent.detail.hash)
    }

    window.addEventListener('hashchange', onHashChange)
    window.addEventListener('doc-navigate-hash', onCustomNav)
    return () => {
      window.removeEventListener('hashchange', onHashChange)
      window.removeEventListener('doc-navigate-hash', onCustomNav)
    }
  }, [data])

  // ✅ Artık tüm early-return'ler hook'lardan SONRA
  if (loading) return <DocSkeleton />

  if (error || !data) {
    return (
      <div className="rounded-2xl border border-destructive/40 bg-destructive/[0.06] p-8 text-center">
        <p className="font-mono text-sm text-destructive">
          {t('İçerik yüklenemedi', 'Failed to load content')}: {error?.message}
        </p>
      </div>
    )
  }

  const { meta, sections } = data

  return (
    <article className="space-y-6">
      <PageHeader
        eyebrow={meta.eyebrow[lang]}
        title={meta.title[lang]}
        description={meta.description[lang]}
      />

      <Reveal className="rounded-2xl border border-border/80 glass-panel surface-shine p-5">
        <span className="font-mono text-xs font-bold uppercase tracking-wider text-primary">
          {t('İÇİNDEKİLER', 'TABLE OF CONTENTS')}
        </span>
        <div className="mt-3 grid grid-cols-1 gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3">
          {sections.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              className="rounded-lg px-2 py-1.5 text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary"
            >
              {section.title[lang]}
            </a>
          ))}
        </div>
      </Reveal>

      {
        sections.map((section, idx) => (
          <Reveal key={section.id} delay={idx * 40}>
            <DocSection id={section.id} title={section.title[lang]}>
              {section.intro && (
                <p>
                  <T value={section.intro} />
                </p>
              )}
              {section.blocks && <BlockList blocks={section.blocks} />}
            </DocSection>

            {section.entries?.map((entry) => (
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
          </Reveal>
        ))
      }

      {
        next && (
          <div className="flex items-center justify-between border-t border-border/80 pt-6">
            <Link
              href={next.href}
              className="inline-flex items-center gap-2 text-sm font-semibold text-primary hover:underline"
            >
              {next.label[lang]}
              <ArrowRight className="size-4" />
            </Link>
          </div>
        )
      }
    </article >
  )
}
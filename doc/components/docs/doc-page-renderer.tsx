'use client'

import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { PageHeader, DocSection } from '@/components/docs/doc-ui'
import { BlockList, T } from '@/components/docs/doc-blocks'
import { ApiEntry } from '@/components/docs/api-entry'
import { Reveal } from '@/components/reveal'
import type { DocPageData } from '@/lib/types/doc'
import { useLanguage } from '@/lib/hooks/use-language'

/** Skeleton shown while the JSON content is loading. */
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

/** Generic documentation page renderer driven by API JSON content. */
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

      {/* Syllabus / TOC quick bar */}
      <Reveal className="rounded-xl border border-border/80 bg-card/40 p-4 backdrop-blur-md">
        <span className="font-mono text-xs font-bold uppercase tracking-wider text-primary">
          {t('İÇİNDEKİLER', 'TABLE OF CONTENTS')}
        </span>
        <div className="mt-2.5 grid grid-cols-1 gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3">
          {sections.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              className="text-muted-foreground transition-colors hover:text-primary"
            >
              {section.title[lang]}
            </a>
          ))}
        </div>
      </Reveal>

      {sections.map((section, idx) => (
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
      ))}

      {next && (
        <div className="flex items-center justify-between border-t border-border/80 pt-6">
          <Link
            href={next.href}
            className="inline-flex items-center gap-2 text-sm font-semibold text-primary hover:underline"
          >
            {next.label[lang]}
            <ArrowRight className="size-4" />
          </Link>
        </div>
      )}
    </article>
  )
}

'use client'

import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'
import { ApiEntry } from '@/components/docs/api-entry'
import { DocSection, PageHeader, Callout } from '@/components/docs/doc-ui'
import { T } from '@/components/docs/doc-blocks'
import { Reveal } from '@/components/reveal'
import type { DocPageData } from '@/lib/types/doc'
import { useLanguage } from '@/lib/hooks/use-language'
import { cn } from '@/lib/utils'

/**
 * Searchable, filterable API reference. The entries themselves come from
 * /api/docs/api-reference (JSON), keeping this component purely presentational.
 */
export function ApiReferenceView({ data }: { data: DocPageData | null }) {
  const { lang, t } = useLanguage()
  const [searchTerm, setSearchTerm] = useState('')
  const [activeCategory, setActiveCategory] = useState<string>('all')

  const sections = data?.sections ?? []

  const categories = useMemo(
    () => [
      { id: 'all', label: { tr: 'Tümü', en: 'All' } },
      ...sections.map((s) => ({ id: s.id, label: s.title })),
    ],
    [sections],
  )

  const filtered = useMemo(() => {
    const term = searchTerm.trim().toLowerCase()
    return sections
      .filter((s) => activeCategory === 'all' || s.id === activeCategory)
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
  }, [sections, activeCategory, searchTerm, lang])

  if (!data) return null

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

      {/* Search + category filter */}
      <div className="sticky top-16 z-30 rounded-2xl border border-border/80 bg-background/80 p-4 shadow-lg backdrop-blur-xl">
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder={t(
              'Fonksiyon veya sınıf ara (örn: matmul, relu_, svd, Variable)...',
              'Search function or class (e.g. matmul, relu_, svd, Variable)...',
            )}
            className="w-full rounded-xl border border-border/80 bg-black/40 py-2 pl-10 pr-4 font-mono text-sm text-foreground placeholder:text-muted-foreground/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {categories.map((cat) => (
            <button
              key={cat.id}
              type="button"
              onClick={() => setActiveCategory(cat.id)}
              className={cn(
                'rounded-lg px-2.5 py-1 font-mono text-xs font-medium transition-all',
                activeCategory === cat.id
                  ? 'border border-primary/50 bg-primary/15 text-primary shadow-[0_0_12px_rgba(74,222,128,0.2)]'
                  : 'border border-transparent text-muted-foreground hover:bg-white/[0.04] hover:text-foreground',
              )}
            >
              {cat.label[lang]}
            </button>
          ))}
        </div>
      </div>

      {filtered.map((section, idx) => (
        <Reveal key={section.id} delay={idx * 30}>
          <DocSection id={section.id} title={section.title[lang]}>
            {section.intro && <T value={section.intro} />}
          </DocSection>

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
        </Reveal>
      ))}

      {filtered.length === 0 && (
        <div className="rounded-2xl border border-border/80 bg-card/40 p-10 text-center font-mono text-sm text-muted-foreground">
          {t('Sonuç bulunamadı', 'No results found')}
        </div>
      )}
    </article>
  )
}

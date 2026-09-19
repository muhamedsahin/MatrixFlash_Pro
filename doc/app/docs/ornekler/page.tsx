'use client'

import Link from 'next/link'
import { DocPageRenderer } from '@/components/docs/doc-page-renderer'
import { useDocContent } from '@/lib/hooks/use-doc-content'
import { site } from '@/lib/site'
import { useLanguage } from '@/lib/hooks/use-language'
import type { DocPageData } from '@/lib/types/doc'

/** Examples page — content is served by /api/docs/examples. */
export default function OrneklerPage() {
  const { t } = useLanguage()
  const { data, loading, error } = useDocContent<DocPageData>('/api/docs/examples')

  return (
    <>
      <DocPageRenderer
        data={data}
        loading={loading}
        error={error}
        next={{
          href: '/docs/api',
          label: { tr: 'Sonraki: API Referansı', en: 'Next: API Reference' },
        }}
      />

      {/* GitHub CTA */}
      <div className="mt-12 rounded-2xl border border-primary/30 bg-gradient-to-b from-primary/[0.08] to-transparent p-8 text-center shadow-xl">
        <h3 className="fluid-section-title font-display text-foreground">
          {t('Daha Fazla Örnek & Test Kodu', 'Looking for More Examples & Benchmarks?')}
        </h3>
        <p className="mx-auto mt-2 max-w-xl text-sm text-muted-foreground">
          {t(
            'Deponun examples/ ve tests/ klasörlerinde cuSOLVER, autograd ve CNN test senaryolarını detaylıca inceleyebilirsiniz.',
            'Explore the examples/ and tests/ directories in the GitHub repository for additional cuSOLVER and Autograd test suites.',
          )}
        </p>
        <a
          href={site.github}
          target="_blank"
          rel="noreferrer"
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition-all hover:scale-105"
        >
          GitHub Repository
        </a>
      </div>

      <div className="mt-8 text-center">
        <Link href="/" className="font-mono text-xs text-muted-foreground hover:text-primary">
          {t('← Ana sayfaya dön', '← Back to home')}
        </Link>
      </div>
    </>
  )
}

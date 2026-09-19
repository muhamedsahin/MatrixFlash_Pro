'use client'

import { DocPageRenderer } from '@/components/docs/doc-page-renderer'
import { useDocContent } from '@/lib/hooks/use-doc-content'
import type { DocPageData } from '@/lib/types/doc'

/** Getting-started page — content is served by /api/docs/getting-started. */
export default function DocsHomePage() {
  const { data, loading, error } = useDocContent<DocPageData>(
    '/api/docs/getting-started',
  )

  return (
    <DocPageRenderer
      data={data}
      loading={loading}
      error={error}
      next={{
        href: '/docs/matris-nedir',
        label: { tr: 'Sonraki: Matris Dersleri', en: 'Next: Matrix Math Course' },
      }}
    />
  )
}

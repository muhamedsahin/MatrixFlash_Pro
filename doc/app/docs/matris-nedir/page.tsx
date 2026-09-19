'use client'

import { DocPageRenderer } from '@/components/docs/doc-page-renderer'
import { useDocContent } from '@/lib/hooks/use-doc-content'
import type { CourseData } from '@/lib/types/doc'

/**
 * Matrix mathematics course — university-level content served by
 * /api/matrix-course and rendered by the generic block renderer.
 */
export default function MatrisDersleriPage() {
  const { data, loading, error } = useDocContent<CourseData>('/api/matrix-course')

  return (
    <DocPageRenderer
      data={data}
      loading={loading}
      error={error}
      next={{
        href: '/docs/api',
        label: { tr: 'Sonraki: API Referansı', en: 'Next: API Reference' },
      }}
    />
  )
}

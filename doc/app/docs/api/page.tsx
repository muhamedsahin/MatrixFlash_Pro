'use client'

import { ApiReferenceView } from '@/components/docs/api-reference-view'
import { useDocContent } from '@/lib/hooks/use-doc-content'
import type { DocPageData } from '@/lib/types/doc'

/** API reference page — content is served by /api/docs/api-reference. */
export default function ApiPage() {
  const { data, loading, error } = useDocContent<DocPageData>(
    '/api/docs/api-reference',
  )

  if (error || (!data && !loading)) {
    return (
      <div className="rounded-2xl border border-destructive/40 bg-destructive/[0.06] p-8 text-center font-mono text-sm text-destructive">
        API içeriği yüklenemedi: {error?.message ?? 'unknown error'}
      </div>
    )
  }

  return (
    <>
      {loading && (
        <div className="mb-6 flex items-center gap-3 font-mono text-xs text-muted-foreground">
          <span className="size-1.5 animate-ping rounded-full bg-primary" />
          API referansı yükleniyor...
        </div>
      )}
      <ApiReferenceView data={data} />
    </>
  )
}

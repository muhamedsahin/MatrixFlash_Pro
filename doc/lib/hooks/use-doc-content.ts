'use client'

import { useEffect, useState } from 'react'
import { fetchJson } from '@/lib/utils/fetch-json'
import type { DocPageData } from '@/lib/types/doc'

export type DocContentState<T> = {
  data: T | null
  loading: boolean
  error: Error | null
}

/**
 * Fetches a documentation payload from the JSON API (e.g. `/api/docs/api-refe
 * rence`) with caching. Pages stay tiny: they only consume the returned state.
 */
export function useDocContent<T = DocPageData>(url: string): DocContentState<T> {
  const [state, setState] = useState<DocContentState<T>>({
    data: null,
    loading: true,
    error: null,
  })

  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true, error: null }))

    fetchJson<T>(url)
      .then((data) => {
        if (alive) setState({ data, loading: false, error: null })
      })
      .catch((err: unknown) => {
        if (alive) {
          setState({
            data: null,
            loading: false,
            error: err instanceof Error ? err : new Error(String(err)),
          })
        }
      })

    return () => {
      alive = false
    }
  }, [url])

  return state
}

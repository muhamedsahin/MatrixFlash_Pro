'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'

export type Language = 'tr' | 'en'

interface LanguageContextType {
  lang: Language
  setLang: (lang: Language) => void
  t: (tr: string, en: string) => string
}

const LanguageContext = createContext<LanguageContextType>({
  lang: 'tr',
  setLang: () => {},
  t: (tr) => tr,
})

function readCookie(name: string): string | null {
  if (typeof document === 'undefined') return null
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`))
  return match ? decodeURIComponent(match[1]) : null
}

function writeCookie(name: string, value: string) {
  try {
    document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${60 * 60 * 24 * 365}; samesite=lax`
  } catch {
    // ignore
  }
}

/** Soft fallback when no geo header is available (local / some hosts). */
function inferLangFromEnvironment(): Language {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || ''
    // Local/dev without CDN geo: treat Istanbul timezone as Turkey.
    if (tz === 'Europe/Istanbul') return 'tr'
  } catch {
    // ignore
  }
  // Anywhere else (or unknown) → English, matching “outside Turkey” rule.
  return 'en'
}

async function resolveInitialLang(): Promise<Language> {
  // 1) Explicit user choice
  try {
    const stored = localStorage.getItem('mfp_lang') as Language | null
    if (stored === 'tr' || stored === 'en') return stored
  } catch {
    // ignore
  }

  const prefCookie = readCookie('mfp_lang')
  if (prefCookie === 'tr' || prefCookie === 'en') return prefCookie

  // 2) Geo cookie set by middleware (Vercel / Cloudflare)
  const geoCookie = readCookie('mfp_geo_lang')
  if (geoCookie === 'tr' || geoCookie === 'en') return geoCookie

  // 3) Ask the server (same geo headers)
  try {
    const res = await fetch('/api/locale', { cache: 'no-store' })
    if (res.ok) {
      const data = (await res.json()) as { lang?: Language | null }
      if (data.lang === 'tr' || data.lang === 'en') return data.lang
    }
  } catch {
    // ignore — fall through
  }

  // 4) Timezone / Accept-Language soft signal
  return inferLangFromEnvironment()
}

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Language>('tr')
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const next = await resolveInitialLang()
      if (!cancelled) {
        setLangState(next)
        setReady(true)
        try {
          document.documentElement.lang = next
        } catch {
          // ignore
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!ready) return
    try {
      document.documentElement.lang = lang
    } catch {
      // ignore
    }
  }, [lang, ready])

  const setLang = (newLang: Language) => {
    setLangState(newLang)
    try {
      localStorage.setItem('mfp_lang', newLang)
    } catch {
      // ignore
    }
    writeCookie('mfp_lang', newLang)
  }

  const t = (tr: string, en: string) => (lang === 'tr' ? tr : en)

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  return useContext(LanguageContext)
}

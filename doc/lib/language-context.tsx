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

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Language>('tr')

  useEffect(() => {
    try {
      const stored = localStorage.getItem('mfp_lang') as Language
      if (stored === 'tr' || stored === 'en') {
        setLangState(stored)
      } else {
        const navLang = navigator.language.toLowerCase()
        if (navLang.startsWith('en')) {
          setLangState('en')
        }
      }
    } catch {
      // localStorage unavailable (SSR/sandbox)
    }
  }, [])

  const setLang = (newLang: Language) => {
    setLangState(newLang)
    try {
      localStorage.setItem('mfp_lang', newLang)
    } catch {
      // ignore
    }
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


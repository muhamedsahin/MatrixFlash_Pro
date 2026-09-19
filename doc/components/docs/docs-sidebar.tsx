'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { docsNav } from '@/lib/site'
import { useLanguage } from '@/lib/language-context'

export function DocsSidebar({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname()
  const { lang } = useLanguage()
  const [currentHash, setCurrentHash] = useState('')

  useEffect(() => {
    const handleHashSync = () => {
      setCurrentHash(window.location.hash.replace('#', ''))
    }
    handleHashSync()
    window.addEventListener('hashchange', handleHashSync)
    window.addEventListener('doc-navigate-hash', ((e: CustomEvent<{ hash: string }>) => {
      if (e.detail?.hash) setCurrentHash(e.detail.hash)
    }) as EventListener)

    return () => {
      window.removeEventListener('hashchange', handleHashSync)
    }
  }, [])

  const handleLinkClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    onNavigate?.()
    const [base, hash] = href.split('#')
    if (hash && pathname === base) {
      e.preventDefault()
      setCurrentHash(hash)
      window.history.replaceState(null, '', `#${hash}`)
      window.dispatchEvent(new CustomEvent('doc-navigate-hash', { detail: { hash } }))

      const el = document.getElementById(hash)
      if (el) {
        const yOffset = -90
        const y = el.getBoundingClientRect().top + window.pageYOffset + yOffset
        window.scrollTo({ top: y, behavior: 'smooth' })
        el.classList.remove('section-highlight')
        void el.offsetWidth // trigger reflow
        el.classList.add('section-highlight')
      }
    }
  }

  return (
    <nav className="flex flex-col gap-6">
      {docsNav.map((section, sIdx) => (
        <div key={sIdx}>
          <h4 className="mb-2 px-3 font-mono text-[11px] uppercase tracking-widest text-primary/80 font-semibold">
            {section.title[lang]}
          </h4>
          <ul className="space-y-0.5">
            {section.items.map((item, iIdx) => {
              const [base, hash] = item.href.split('#')
              const isSamePage = pathname === base
              const isActive =
                isSamePage &&
                (hash ? currentHash === hash : !currentHash && !item.href.includes('#'))

              return (
                <li key={iIdx}>
                  <Link
                    href={item.href}
                    onClick={(e) => handleLinkClick(e, item.href)}
                    className={cn(
                      'group flex items-center justify-between rounded-lg border-l-2 px-3 py-1.5 text-sm transition-all duration-150',
                      isActive
                        ? 'border-primary bg-primary/10 font-medium text-foreground glow-primary shadow-[0_0_12px_rgba(74,222,128,0.15)]'
                        : 'border-transparent text-muted-foreground hover:border-border/80 hover:bg-white/[0.03] hover:text-foreground',
                    )}
                  >
                    <span className="truncate">{item.label[lang]}</span>
                    {item.badge && (
                      <span className="ml-2 shrink-0 rounded-full border border-primary/30 bg-primary/15 px-1.5 py-0.5 font-mono text-[9px] text-primary">
                        {item.badge}
                      </span>
                    )}
                  </Link>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
    </nav>
  )
}

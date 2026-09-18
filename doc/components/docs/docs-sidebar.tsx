'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { docsNav } from '@/lib/site'
import { useLanguage } from '@/lib/language-context'

export function DocsSidebar({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname()
  const { lang } = useLanguage()

  return (
    <nav className="flex flex-col gap-6">
      {docsNav.map((section, sIdx) => (
        <div key={sIdx}>
          <h4 className="mb-2 px-3 font-mono text-[11px] uppercase tracking-widest text-primary/80 font-semibold">
            {section.title[lang]}
          </h4>
          <ul className="space-y-0.5">
            {section.items.map((item, iIdx) => {
              const [base] = item.href.split('#')
              const active =
                pathname === base &&
                (!item.href.includes('#') ||
                  item.href === base ||
                  section.items.filter((s) => s.href.split('#')[0] === base)
                    .length === 1)
              return (
                <li key={iIdx}>
                  <Link
                    href={item.href}
                    onClick={onNavigate}
                    className={cn(
                      'group flex items-center justify-between rounded-lg border-l-2 px-3 py-1.5 text-sm transition-all duration-150',
                      pathname === base && !item.href.includes('#')
                        ? 'border-primary bg-primary/10 font-medium text-foreground glow-primary'
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

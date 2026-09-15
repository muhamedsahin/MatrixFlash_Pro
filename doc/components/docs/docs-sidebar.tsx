'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '../../lib/utils'
import { docsNav } from '../../lib/site'

export function DocsSidebar({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname()

  return (
    <nav className="flex flex-col gap-7">
      {docsNav.map((section) => (
        <div key={section.title}>
          <h4 className="mb-2 px-3 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
            {section.title}
          </h4>
          <ul className="space-y-0.5">
            {section.items.map((item) => {
              const [base] = item.href.split('#')
              const active =
                pathname === base &&
                (!item.href.includes('#') ||
                  item.href === base ||
                  section.items.filter((s) => s.href.split('#')[0] === base)
                    .length === 1)
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    onClick={onNavigate}
                    className={cn(
                      'flex items-center gap-2 rounded-md border-l-2 px-3 py-1.5 text-sm transition-colors',
                      pathname === base && !item.href.includes('#')
                        ? 'border-primary bg-primary/5 text-foreground'
                        : 'border-transparent text-muted-foreground hover:border-border hover:text-foreground',
                    )}
                  >
                    {item.label}
                    {item.badge && (
                      <span className="ml-auto rounded bg-primary/15 px-1.5 py-0.5 font-mono text-[10px] text-primary">
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

import type { ReactNode } from 'react'
import { SiteNav } from '../../components/site-nav'
import { SiteFooter } from '../../components/site-footer'
import { DocsSidebar } from '../../components/docs/docs-sidebar'
import { DocsMobileNav } from '../../components/docs/docs-mobile-nav'

export default function DocsLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <SiteNav />
      <div className="mx-auto flex max-w-7xl gap-10 px-4 pt-16 sm:px-6 lg:px-8">
        <aside className="sticky top-16 hidden h-[calc(100vh-4rem)] w-60 shrink-0 overflow-y-auto py-10 lg:block">
          <DocsSidebar />
        </aside>

        <div className="min-w-0 flex-1 py-10">
          <div className="mb-6 lg:hidden">
            <DocsMobileNav />
          </div>
          <div className="max-w-3xl">{children}</div>
        </div>
      </div>
      <SiteFooter />
    </div>
  )
}

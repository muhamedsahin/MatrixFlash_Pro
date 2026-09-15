import type { ReactNode } from 'react'
import { Info, Lightbulb, TriangleAlert } from 'lucide-react'
import { cn } from '../../lib/utils'

export function PageHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string
  title: string
  description: string
}) {
  return (
    <header className="mb-10 border-b border-border pb-8">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-primary">
        {eyebrow}
      </p>
      <h1 className="mt-3 text-balance text-4xl font-bold tracking-tight sm:text-5xl">
        {title}
      </h1>
      <p className="mt-4 max-w-2xl text-pretty text-lg leading-relaxed text-muted-foreground">
        {description}
      </p>
    </header>
  )
}

export function DocSection({
  id,
  title,
  children,
}: {
  id?: string
  title: string
  children: ReactNode
}) {
  return (
    <section id={id} className="scroll-mt-24 py-8">
      <h2 className="group flex items-center gap-2 text-2xl font-bold tracking-tight">
        {id ? (
          <a href={`#${id}`} className="flex items-center gap-2">
            {title}
            <span className="font-mono text-primary opacity-0 transition-opacity group-hover:opacity-100">
              #
            </span>
          </a>
        ) : (
          title
        )}
      </h2>
      <div className="mt-5 space-y-4 leading-relaxed text-muted-foreground">
        {children}
      </div>
    </section>
  )
}

export function SubHeading({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <h3
      id={id}
      className="scroll-mt-24 pt-4 text-lg font-semibold text-foreground"
    >
      {children}
    </h3>
  )
}

const calloutStyles = {
  info: {
    icon: Info,
    cls: 'border-accent/30 bg-accent/[0.06] text-accent',
  },
  tip: {
    icon: Lightbulb,
    cls: 'border-primary/30 bg-primary/[0.06] text-primary',
  },
  warn: {
    icon: TriangleAlert,
    cls: 'border-chart-4/30 bg-chart-4/[0.06] text-chart-4',
  },
}

export function Callout({
  type = 'info',
  title,
  children,
}: {
  type?: 'info' | 'tip' | 'warn'
  title?: string
  children: ReactNode
}) {
  const { icon: Icon, cls } = calloutStyles[type]
  return (
    <div className={cn('my-6 rounded-xl border px-5 py-4', cls)}>
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 size-5 shrink-0" />
        <div className="text-sm leading-relaxed text-foreground">
          {title && <p className="mb-1 font-semibold">{title}</p>}
          <div className="text-muted-foreground [&_code]:font-mono [&_code]:text-primary">
            {children}
          </div>
        </div>
      </div>
    </div>
  )
}

export function InlineCode({ children }: { children: ReactNode }) {
  return (
    <code className="rounded bg-secondary/70 px-1.5 py-0.5 font-mono text-[0.85em] text-primary">
      {children}
    </code>
  )
}

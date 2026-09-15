import type { ReactNode } from 'react'
import { CodeBlock } from '../code-block'

export type ApiParam = { name: string; type: string; desc: string }

export function ApiEntry({
  name,
  signature,
  badge,
  children,
  params,
  returns,
  example,
}: {
  name: string
  signature: string
  badge?: string
  children: ReactNode
  params?: ApiParam[]
  returns?: string
  example?: string
}) {
  const id = name.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase()
  return (
    <div
      id={id}
      className="scroll-mt-24 rounded-2xl border border-border bg-card/40 p-6"
    >
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="font-mono text-lg font-semibold text-foreground">
          {name}
        </h3>
        {badge && (
          <span className="rounded-full border border-primary/30 bg-primary/10 px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-primary">
            {badge}
          </span>
        )}
      </div>

      <div className="mt-4 overflow-x-auto rounded-lg border border-border bg-[oklch(0.13_0.008_160)] px-4 py-3">
        <code className="font-mono text-[13px] text-accent">{signature}</code>
      </div>

      <div className="mt-4 text-sm leading-relaxed text-muted-foreground [&_code]:rounded [&_code]:bg-secondary/70 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-primary">
        {children}
      </div>

      {params && params.length > 0 && (
        <div className="mt-5">
          <p className="mb-2 font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Parametreler
          </p>
          <div className="divide-y divide-border/60 overflow-hidden rounded-lg border border-border">
            {params.map((p) => (
              <div
                key={p.name}
                className="grid grid-cols-1 gap-1 bg-card/40 px-4 py-3 sm:grid-cols-[160px_1fr]"
              >
                <div className="flex flex-col">
                  <code className="font-mono text-sm text-primary">
                    {p.name}
                  </code>
                  <code className="font-mono text-xs text-muted-foreground">
                    {p.type}
                  </code>
                </div>
                <p className="text-sm text-muted-foreground">{p.desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {returns && (
        <p className="mt-4 text-sm">
          <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Döndürür{' '}
          </span>
          <code className="ml-2 rounded bg-secondary/70 px-1.5 py-0.5 font-mono text-sm text-accent">
            {returns}
          </code>
        </p>
      )}

      {example && (
        <div className="mt-5">
          <CodeBlock code={example} />
        </div>
      )}
    </div>
  )
}

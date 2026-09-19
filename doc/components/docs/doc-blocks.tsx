'use client'

import type { ReactNode } from 'react'
import { CodeBlock } from '@/components/code-block'
import { Math } from '@/components/math'
import { MatrixFigure } from '@/components/docs/matrix-figure'
import { MatrixCalculator } from '@/components/interactive/matrix-calculator'
import {
  Callout,
  InlineCode,
  SubHeading,
} from '@/components/docs/doc-ui'
import type { DocBlock } from '@/lib/types/doc'
import { useLanguage } from '@/lib/hooks/use-language'

/** Parses light inline markup: `code` spans and **bold** text. */
function RichText({ text }: { text: string }) {
  const segments = text.split(/`([^`]+)`/g)
  return (
    <>
      {segments.map((seg, i) => {
        if (i % 2 === 1) return <InlineCode key={i}>{seg}</InlineCode>
        const boldParts = seg.split(/\*\*([^*]+)\*\*/g)
        return (
          <span key={i}>
            {boldParts.map((part, j) =>
              j % 2 === 1 ? (
                <strong key={j} className="font-semibold text-foreground">
                  {part}
                </strong>
              ) : (
                <span key={j}>{part}</span>
              ),
            )}
          </span>
        )
      })}
    </>
  )
}

/** Localized text picker. */
export function T({ value }: { value: LText }) {
  const { lang } = useLanguage()
  return <>{value[lang]}</>
}

function BlockNode({ block }: { block: DocBlock }): ReactNode {
  const { lang } = useLanguage()

  switch (block.type) {
    case 'p':
      return (
        <p>
          <RichText text={block.text[lang]} />
        </p>
      )
    case 'h3':
      return <SubHeading>{block.text[lang]}</SubHeading>
    case 'code':
      return (
        <CodeBlock code={block.code} lang={block.lang} filename={block.filename} />
      )
    case 'callout':
      return (
        <Callout type={block.variant} title={block.title?.[lang]}>
          <RichText text={block.text[lang]} />
        </Callout>
      )
    case 'math':
      return <Math display={block.display ?? true}>{block.tex}</Math>
    case 'list': {
      const items = block.items.map((item, i) => (
        <li key={i} className="leading-relaxed">
          <RichText text={item[lang]} />
        </li>
      ))
      return block.ordered ? (
        <ol className="list-decimal space-y-2 pl-6 marker:font-mono marker:text-primary">
          {items}
        </ol>
      ) : (
        <ul className="list-disc space-y-2 pl-6 marker:text-primary">{items}</ul>
      )
    }
    case 'table':
      return (
        <div className="my-5 overflow-x-auto rounded-xl border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-card/60 font-mono text-xs uppercase tracking-wider text-primary">
              <tr>
                {block.headers.map((h, i) => (
                  <th key={i} className="border-b border-border px-4 py-3 font-semibold">
                    {h[lang]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {block.rows.map((row, r) => (
                <tr key={r} className="transition-colors hover:bg-white/[0.02]">
                  {row.map((cell, c) => (
                    <td key={c} className="px-4 py-2.5 text-muted-foreground">
                      <RichText text={cell[lang]} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    case 'figure':
      return (
        <MatrixFigure
          data={block.data}
          caption={block.caption?.[lang]}
          highlight={block.highlight ?? []}
          accent={block.accent}
        />
      )
    case 'steps':
      return (
        <div className="my-5 space-y-3">
          {block.items.map((step, i) => (
            <div
              key={i}
              className="flex gap-4 rounded-xl border border-border/80 bg-card/40 p-4"
            >
              <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-primary/40 bg-primary/10 font-mono text-sm font-bold text-primary">
                {i + 1}
              </span>
              <div className="min-w-0">
                <p className="font-semibold text-foreground">{step.title[lang]}</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  <RichText text={step.text[lang]} />
                </p>
              </div>
            </div>
          ))}
        </div>
      )
    case 'simulator':
      return <MatrixCalculator />
    default:
      return null
  }
}

/** Renders a list of content blocks. */
export function BlockList({ blocks }: { blocks: DocBlock[] }) {
  return (
    <>
      {blocks.map((block, i) => (
        <BlockNode key={i} block={block} />
      ))}
    </>
  )
}


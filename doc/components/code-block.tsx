'use client'

import { useState, type ReactNode } from 'react'
import { Check, Copy, Terminal } from 'lucide-react'
import { cn } from '@/lib/utils'

type CodeBlockProps = {
  code: string
  lang?: 'cpp' | 'bash' | 'text'
  filename?: string
  className?: string
}

const CPP_KEYWORDS =
  /\b(auto|int|float|double|void|bool|char|const|constexpr|return|if|else|for|while|struct|class|public|private|namespace|using|include|std|std::size_t|size_t|true|false|nullptr|this|new|delete|template|typename)\b/g

function highlight(line: string, lang: CodeBlockProps['lang']): ReactNode[] {
  if (lang === 'bash') {
    if (line.trim().startsWith('#')) {
      return [
        <span key="c" className="text-muted-foreground/70">
          {line}
        </span>,
      ]
    }
    const parts = line.split(/(\s+)/)
    return parts.map((p, i) => {
      if (i === 0 && p.trim())
        return (
          <span key={i} className="text-primary">
            {p}
          </span>
        )
      if (p.startsWith('--') || p.startsWith('-'))
        return (
          <span key={i} className="text-accent">
            {p}
          </span>
        )
      return <span key={i}>{p}</span>
    })
  }

  if (lang === 'text') return [<span key="t">{line}</span>]

  // cpp
  const nodes: ReactNode[] = []
  // comments
  const commentIdx = line.indexOf('//')
  let work = line
  let comment: string | null = null
  if (commentIdx !== -1) {
    comment = line.slice(commentIdx)
    work = line.slice(0, commentIdx)
  }

  // tokenize by strings first
  const tokenRegex = /("[^"]*"|'[^']*')/g
  const segments = work.split(tokenRegex)
  segments.forEach((seg, si) => {
    if (!seg) return
    if (/^["'].*["']$/.test(seg)) {
      nodes.push(
        <span key={`s${si}`} className="text-chart-4">
          {seg}
        </span>,
      )
      return
    }
    // numbers + keywords + calls
    const inner = seg.split(
      /(\b\d+\.?\d*f?\b|\bMatrix\b|\.[a-zA-Z_]+(?=\()|::[a-zA-Z_]+)/g,
    )
    inner.forEach((piece, pi) => {
      if (!piece) return
      if (/^\b\d+\.?\d*f?\b$/.test(piece)) {
        nodes.push(
          <span key={`s${si}n${pi}`} className="text-accent">
            {piece}
          </span>,
        )
        return
      }
      if (piece === 'Matrix') {
        nodes.push(
          <span key={`s${si}m${pi}`} className="text-chart-3 font-medium">
            {piece}
          </span>,
        )
        return
      }
      if (/^\.[a-zA-Z_]+$/.test(piece) || /^::[a-zA-Z_]+$/.test(piece)) {
        nodes.push(
          <span key={`s${si}c${pi}`} className="text-primary">
            {piece}
          </span>,
        )
        return
      }
      // Split with a capture group so each keyword is rendered exactly once.
      const keywordParts = piece.split(CPP_KEYWORDS)
      const keywordPattern = new RegExp(`^${CPP_KEYWORDS.source}$`)
      const out: ReactNode[] = keywordParts.map((part, ki) =>
        keywordPattern.test(part) ? (
          <span key={`kw${ki}`} className="text-chart-3">
            {part}
          </span>
        ) : (
          <span key={`k${ki}`}>{part}</span>
        ),
      )
      nodes.push(<span key={`s${si}w${pi}`}>{out}</span>)
    })
  })

  if (comment)
    nodes.push(
      <span key="cmt" className="text-muted-foreground/60 italic">
        {comment}
      </span>,
    )

  return nodes
}

export function CodeBlock({ code, lang = 'cpp', filename, className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const lines = code.replace(/\n$/, '').split('\n')

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* ignore */
    }
  }

  return (
    <div
      className={cn(
        'group relative overflow-hidden rounded-xl border border-border bg-[oklch(0.13_0.008_160)]',
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-border/70 bg-card/50 px-4 py-2.5">
        <div className="flex items-center gap-2 font-mono text-xs text-muted-foreground">
          {lang === 'bash' ? (
            <Terminal className="size-3.5 text-primary" />
          ) : (
            <span className="flex gap-1.5">
              <span className="size-2.5 rounded-full bg-destructive/70" />
              <span className="size-2.5 rounded-full bg-chart-4/70" />
              <span className="size-2.5 rounded-full bg-primary/70" />
            </span>
          )}
          <span className="ml-1">{filename ?? (lang === 'bash' ? 'terminal' : 'main.cpp')}</span>
        </div>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 font-mono text-xs text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          aria-label="Kodu kopyala"
        >
          {copied ? (
            <>
              <Check className="size-3.5 text-primary" /> Kopyalandı
            </>
          ) : (
            <>
              <Copy className="size-3.5" /> Kopyala
            </>
          )}
        </button>
      </div>
      <div className="overflow-x-auto">
        <pre className="min-w-full py-4 font-mono text-[13px] leading-relaxed">
          <code>
            {lines.map((line, i) => (
              <div key={i} className="flex px-4 hover:bg-white/[0.02]">
                <span className="mr-4 w-6 shrink-0 select-none text-right text-muted-foreground/40">
                  {i + 1}
                </span>
                <span className="whitespace-pre">
                  {line ? highlight(line, lang) : '\u00a0'}
                </span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  )
}

import 'katex/dist/katex.min.css'
import katex from 'katex'
import { cn } from '@/lib/utils'

export function Math({
  children,
  display = false,
  className,
}: {
  children: string
  display?: boolean
  className?: string
}) {
  const html = katex.renderToString(children, {
    displayMode: display,
    throwOnError: false,
    trust: false,
    output: 'htmlAndMathml',
  })

  if (display) {
    return (
      <div
        className={cn(
          'my-6 overflow-x-auto rounded-xl border border-border bg-card/50 px-6 py-5 text-center text-foreground',
          className,
        )}
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }

  return (
    <span
      className={cn('text-foreground', className)}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

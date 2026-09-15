import { cn } from '../../lib/utils'

type MatrixFigureProps = {
  data: (string | number)[][]
  caption?: string
  highlight?: [number, number][]
  accent?: 'primary' | 'accent' | 'chart-3'
  className?: string
}

/** Static matrix rendered with bracket borders, for math explanations. */
export function MatrixFigure({
  data,
  caption,
  highlight = [],
  accent = 'primary',
  className,
}: MatrixFigureProps) {
  const cols = data[0]?.length ?? 0
  const isHi = (r: number, c: number) =>
    highlight.some(([hr, hc]) => hr === r && hc === c)

  const accentCls =
    accent === 'accent'
      ? 'bg-accent/20 text-accent ring-accent/40'
      : accent === 'chart-3'
        ? 'bg-chart-3/20 text-chart-3 ring-chart-3/40'
        : 'bg-primary/20 text-primary ring-primary/40'

  return (
    <figure className={cn('my-6 flex flex-col items-center', className)}>
      <div className="relative flex items-stretch gap-1 px-3">
        <span className="w-2 rounded-l-md border-y-2 border-l-2 border-muted-foreground/50" />
        <div
          className="grid gap-1.5 py-2"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        >
          {data.map((row, r) =>
            row.map((val, c) => (
              <span
                key={`${r}-${c}`}
                className={cn(
                  'grid h-11 w-14 place-items-center rounded-md font-mono text-sm tabular-nums ring-1 transition-colors',
                  isHi(r, c)
                    ? accentCls
                    : 'bg-secondary/50 text-foreground ring-border',
                )}
              >
                {val}
              </span>
            )),
          )}
        </div>
        <span className="w-2 rounded-r-md border-y-2 border-r-2 border-muted-foreground/50" />
      </div>
      {caption && (
        <figcaption className="mt-3 text-center font-mono text-xs text-muted-foreground">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}

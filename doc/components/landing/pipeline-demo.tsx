'use client'

import { useEffect, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Reveal } from '@/components/reveal'

type Stage = {
  key: string
  call: string
  desc: string
  grid: string[]
}

const STAGES: Stage[] = [
  {
    key: 'input',
    call: 'Matrix input',
    desc: 'Ham veri device belleğine yüklenir',
    grid: ['1.00', '-2.0', '3.00', '4.00'],
  },
  {
    key: 'matmul',
    call: 'input * weights',
    desc: 'Shared-memory tiled CUDA matris çarpımı',
    grid: ['0.90', '-1.7', '2.80', '3.60'],
  },
  {
    key: 'relu',
    call: '.relu()',
    desc: 'Negatif değerler sıfırlanır (max(0, x))',
    grid: ['0.90', '0.00', '2.80', '3.60'],
  },
  {
    key: 'softmax',
    call: '.softmax()',
    desc: 'Sayısal kararlı olasılık dağılımı',
    grid: ['0.08', '0.03', '0.30', '0.59'],
  },
]

export function PipelineDemo() {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setStep((s) => (s + 1) % STAGES.length), 1800)
    return () => clearInterval(id)
  }, [])

  return (
    <section className="relative border-y border-border bg-card/30 py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Reveal className="mx-auto max-w-2xl text-center">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-primary">
            Zincirleme API
          </p>
          <h2 className="mt-4 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            İşlemler device belleğinde akar
          </h2>
          <p className="mt-4 text-pretty text-muted-foreground">
            Veri host tarafına yalnızca <code className="font-mono text-primary">download()</code>{' '}
            çağrıldığında aktarılır. Gereksiz host↔device transferi ortadan kalkar.
          </p>
        </Reveal>

        <Reveal className="mt-14" delay={100}>
          <div className="flex flex-col items-stretch gap-3 lg:flex-row lg:items-center lg:justify-center">
            {STAGES.map((stage, i) => (
              <div key={stage.key} className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setStep(i)}
                  className={cn(
                    'w-full min-w-[200px] rounded-xl border p-4 text-left transition-all duration-500 lg:w-auto',
                    i === step
                      ? 'border-primary/50 bg-primary/[0.07] glow-primary'
                      : 'border-border bg-card/60 opacity-60',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <code
                      className={cn(
                        'font-mono text-sm font-medium',
                        i === step ? 'text-primary' : 'text-foreground',
                      )}
                    >
                      {stage.call}
                    </code>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      0{i + 1}
                    </span>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-1">
                    {stage.grid.map((v, gi) => (
                      <span
                        key={gi}
                        className={cn(
                          'grid h-8 place-items-center rounded font-mono text-xs tabular-nums transition-colors duration-500',
                          i === step
                            ? 'bg-primary/15 text-primary'
                            : 'bg-secondary/40 text-muted-foreground',
                        )}
                      >
                        {v}
                      </span>
                    ))}
                  </div>
                  <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
                    {stage.desc}
                  </p>
                </button>
                {i < STAGES.length - 1 && (
                  <ArrowRight
                    className={cn(
                      'hidden size-5 shrink-0 transition-colors duration-500 lg:block',
                      i < step ? 'text-primary' : 'text-muted-foreground/40',
                    )}
                  />
                )}
              </div>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  )
}

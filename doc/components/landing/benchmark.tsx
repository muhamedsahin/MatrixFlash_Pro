import { Reveal } from '@/components/reveal'
import { CodeBlock } from '@/components/code-block'

const stats = [
  { value: '30+', label: 'GPU operasyonu' },
  { value: '8', label: 'modüler .cu dosyası' },
  { value: '7', label: 'matris oluşturucu' },
  { value: '100%', label: 'device belleğinde zincir' },
]

const benchCode = `# Benchmark aracını 1024x1024 boyutunda çalıştır
& .\\build\\Release\\matrix_pro_benchmark.exe 1024

# Tiled CUDA matris çarpımının GPU üzerindeki
# gerçek zamanlı performansını raporlar.`

export function Benchmark() {
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid gap-14 lg:grid-cols-2 lg:items-center">
          <Reveal>
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-primary">
              Performans
            </p>
            <h2 className="mt-4 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
              GPU üzerinde ölçülebilir hız
            </h2>
            <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
              Proje ile birlikte gelen benchmark aracı, tiled CUDA matris
              çarpımının GPU üzerindeki gerçek zamanlı performansını ölçer.
              Matris boyutunu parametre olarak verebilirsiniz.
            </p>

            <div className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border bg-border">
              {stats.map((s) => (
                <div key={s.label} className="bg-card p-5">
                  <div className="font-mono text-3xl font-bold text-primary text-glow">
                    {s.value}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {s.label}
                  </div>
                </div>
              ))}
            </div>
          </Reveal>

          <Reveal delay={120}>
            <CodeBlock code={benchCode} lang="bash" filename="powershell" />
            <div className="mt-4 space-y-3">
              {[
                { size: '256×256', w: 22 },
                { size: '512×512', w: 45 },
                { size: '1024×1024', w: 78 },
                { size: '2048×2048', w: 100 },
              ].map((b) => (
                <div key={b.size} className="flex items-center gap-3">
                  <span className="w-24 shrink-0 font-mono text-xs text-muted-foreground">
                    {b.size}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary/60">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-primary/60 to-primary"
                      style={{ width: `${b.w}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

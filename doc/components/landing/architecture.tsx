import { FileCode2, FolderTree } from 'lucide-react'
import { Reveal } from '@/components/reveal'

const tree = [
  { path: 'include/matrix_pro/matrix.hpp', desc: 'Public Matrix API tanımı', depth: 1 },
  { path: 'operations.hpp', desc: 'GPU operasyon imzaları', depth: 1 },
  { path: 'cuda_utils.hpp', desc: 'CUDA hata kontrolü', depth: 1 },
  { path: 'src/matrix.cu', desc: 'Yaşam döngüsü, save/load', depth: 0 },
  { path: 'matrix_factories.cu', desc: 'zeros, ones, identity, randn…', depth: 0 },
  { path: 'operations_matmul.cu', desc: 'shared-memory tiled matmul', depth: 0 },
  { path: 'operations_elementwise.cu', desc: 'add, subtract, Hadamard, scalar', depth: 0 },
  { path: 'operations_statistics.cu', desc: 'GPU indirgemeleri', depth: 0 },
  { path: 'operations_advanced.cu', desc: 'determinant & inverse', depth: 0 },
]

export function Architecture() {
  return (
    <section className="relative border-y border-border bg-card/30 py-24">
      <div className="mx-auto grid max-w-7xl gap-14 px-4 sm:px-6 lg:grid-cols-2 lg:items-center lg:px-8">
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-primary">
            Mimari
          </p>
          <h2 className="mt-4 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            Tek sorumluluk, modüler dosya yapısı
          </h2>
          <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
            Her <code className="font-mono text-primary">.cu</code> dosyası tek
            bir sorumluluğa odaklanır. Matris çarpımı mantığı yalnızca{' '}
            <code className="font-mono text-foreground">operations_matmul.cu</code>{' '}
            içinde, istatistiksel indirgemeler ise yalnızca{' '}
            <code className="font-mono text-foreground">operations_statistics.cu</code>{' '}
            içinde bulunur.
          </p>
          <p className="mt-4 text-pretty leading-relaxed text-muted-foreground">
            Bu yapı, kütüphaneye yeni bir operasyon eklemeyi veya mevcut bir
            operasyonu hata ayıklamayı önemli ölçüde kolaylaştırır.
          </p>
          <div className="mt-6 flex items-center gap-2 font-mono text-xs text-muted-foreground">
            <FolderTree className="size-4 text-primary" />
            include/ · src/ · tests/ · benchmarks/ · examples/
          </div>
        </Reveal>

        <Reveal delay={120}>
          <div className="overflow-hidden rounded-xl border border-border bg-[oklch(0.13_0.008_160)]">
            <div className="flex items-center gap-2 border-b border-border/70 bg-card/50 px-4 py-2.5 font-mono text-xs text-muted-foreground">
              <FolderTree className="size-3.5 text-primary" />
              MatrixFlash_Pro/
            </div>
            <ul className="divide-y divide-border/50">
              {tree.map((node) => (
                <li
                  key={node.path}
                  className="flex items-center gap-3 px-4 py-2.5 transition-colors hover:bg-white/[0.02]"
                  style={{ paddingLeft: `${16 + node.depth * 18}px` }}
                >
                  <FileCode2 className="size-3.5 shrink-0 text-primary/70" />
                  <code className="font-mono text-xs text-foreground">
                    {node.path}
                  </code>
                  <span className="ml-auto hidden truncate text-xs text-muted-foreground sm:block">
                    {node.desc}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </Reveal>
      </div>
    </section>
  )
}

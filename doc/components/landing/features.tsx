import {
  Boxes,
  Gauge,
  GitBranch,
  Layers,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { Reveal } from '@/components/reveal'

const features = [
  {
    icon: Gauge,
    title: 'Tiled CUDA performansı',
    body: 'Matris çarpımı shared-memory tiled kernel ile, elementwise işlemler ise özel CUDA çekirdekleriyle GPU üzerinde çalışır.',
  },
  {
    icon: GitBranch,
    title: 'Zincirleme API',
    body: 'İşlemler device belleğinde art arda zincirlenir. Veri host tarafına yalnızca açıkça download() çağrıldığında aktarılır.',
  },
  {
    icon: Layers,
    title: 'Modüler Kod Tabanı',
    body: 'Her operasyon grubu kendi .cu dosyasında ayrıştırılmıştır — okunabilirlik ve bakım kolaylığı ön planda.',
  },
  {
    icon: ShieldCheck,
    title: 'Sayısal Kararlılık',
    body: 'softmax gibi hassas fonksiyonlar, taşma (overflow) sorunlarına karşı kararlı şekilde implemente edilmiştir.',
  },
  {
    icon: Sparkles,
    title: 'Aktivasyonlar',
    body: 'relu, softmax, sigmoid ve tanh doğrudan GPU üzerinde çalışır. Küçük/orta ölçekli sinir ağları için ideal.',
  },
  {
    icon: Boxes,
    title: 'Kapsamlı Araçlar',
    body: 'Davranış testleri, GPU matmul benchmark aracı ve kalıcı veri desteği (save/load) proje ile birlikte gelir.',
  },
]

export function Features() {
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <Reveal className="max-w-2xl">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-primary">
            Neden MatrixFlash-Pro?
          </p>
          <h2 className="mt-4 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            Endüstri standardı performans, sade arayüz
          </h2>
          <p className="mt-4 text-pretty text-muted-foreground">
            Hibrit yaklaşım: kritik işlemler tiled CUDA kernel ile, indirgeme ve
            elementwise operasyonlar özel GPU kernelları ile çalışır.
          </p>
        </Reveal>

        <div className="mt-14 grid gap-px overflow-hidden rounded-2xl border border-border bg-border sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => (
            <Reveal
              key={f.title}
              delay={i * 60}
              className="group relative bg-card p-7 transition-colors hover:bg-card/60"
            >
              <div className="flex size-11 items-center justify-center rounded-lg bg-primary/10 ring-1 ring-primary/20 transition-colors group-hover:bg-primary/20">
                <f.icon className="size-5 text-primary" />
              </div>
              <h3 className="mt-5 text-lg font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {f.body}
              </p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}

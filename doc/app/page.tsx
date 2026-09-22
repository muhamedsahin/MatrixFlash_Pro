import { SiteNav } from '@/components/site-nav'
import { SiteFooter } from '@/components/site-footer'
import { Hero } from '@/components/landing/hero'
import { Quickstart } from '@/components/landing/quickstart'
import { ModuleExplorer } from '@/components/interactive/module-explorer'
import { PipelineDemo } from '@/components/landing/pipeline-demo'
import { Features } from '@/components/landing/features'
import { Architecture } from '@/components/landing/architecture'
import { Benchmark } from '@/components/landing/benchmark'
import { Cta } from '@/components/landing/cta'

export default function HomePage() {
  return (
    <div className="relative min-h-screen bg-background text-foreground selection:bg-primary/20 selection:text-primary">
      <meta name="google-site-verification" content="oBU2n6z8B_IxbjJQ2qmMy2GM8Q16uhgRRtVIXDCqXkg" />
      <div className="pointer-events-none fixed inset-0 -z-10 mesh-aurora opacity-50" aria-hidden />
      <SiteNav />
      <main className="space-y-2">
        <Hero />
        <Quickstart />
        <section className="relative mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <ModuleExplorer />
        </section>
        <PipelineDemo />
        <Features />
        <Architecture />
        <Benchmark />
        <Cta />
      </main>
      <SiteFooter />
    </div>
  )
}


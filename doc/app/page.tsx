import { SiteNav } from '@/components/site-nav'
import { SiteFooter } from '@/components/site-footer'
import { Hero } from '@/components/landing/hero'
import { PipelineDemo } from '@/components/landing/pipeline-demo'
import { Features } from '@/components/landing/features'
import { Architecture } from '@/components/landing/architecture'
import { Benchmark } from '@/components/landing/benchmark'
import { Cta } from '@/components/landing/cta'

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <SiteNav />
      <main>
        <Hero />
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

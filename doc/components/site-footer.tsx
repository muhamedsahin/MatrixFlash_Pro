import Link from 'next/link'
import { GithubIcon } from '@/components/github-icon'
import { site, topNav } from '@/lib/site'
import { Logo } from '@/components/logo'

export function SiteFooter() {
  return (
    <footer className="relative border-t border-border">
      <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
        <div className="flex flex-col justify-between gap-10 md:flex-row">
          <div className="max-w-sm">
            <Link href="/" className="flex items-center gap-2.5">
              <Logo className="size-8" />
              <span className="font-mono text-sm font-bold">
                MatrixFlash<span className="text-primary">-Pro</span>
              </span>
            </Link>
            <p className="mt-4 text-sm text-muted-foreground">
              {site.tagline}. Tiled CUDA matris çarpımı, GPU üzerinde
              indirgemeler ve aktivasyon fonksiyonları.
            </p>
            <a
              href={site.github}
              target="_blank"
              rel="noreferrer"
              className="mt-5 inline-flex items-center gap-2 rounded-md border border-border bg-secondary/50 px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <GithubIcon className="size-4" /> muhamedsahin/MatrixFlash_Pro
            </a>
          </div>

          <div className="grid grid-cols-2 gap-10 sm:grid-cols-3">
            <div>
              <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
                Dokümantasyon
              </h3>
              <ul className="mt-4 space-y-2.5">
                {topNav.slice(1).map((item) => (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      className="text-sm text-muted-foreground transition-colors hover:text-primary"
                    >
                      {item.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
                Teknoloji
              </h3>
              <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground">
                <li>CUDA {site.cuda}</li>
                <li>CUDA kernels</li>
                <li>{site.cpp}</li>
                <li>CMake</li>
              </ul>
            </div>
            <div>
              <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
                Geliştirici
              </h3>
              <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground">
                <li>{site.author}</li>
                <li>Bursa Teknik Üniv.</li>
                <li>Bilgisayar Müh.</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-border pt-6 sm:flex-row">
          <p className="font-mono text-xs text-muted-foreground">
            © {new Date().getFullYear()} MatrixFlash-Pro · {site.author}
          </p>
          <p className="font-mono text-xs text-muted-foreground">
            Yapay zeka desteğiyle geliştirildi
          </p>
        </div>
      </div>
    </footer>
  )
}

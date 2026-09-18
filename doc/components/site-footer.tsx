'use client'

import Link from 'next/link'
import { GithubIcon } from '@/components/github-icon'
import { site, topNav } from '@/lib/site'
import { Logo } from '@/components/logo'
import { useLanguage } from '@/lib/language-context'

export function SiteFooter() {
  const { lang, t } = useLanguage()

  return (
    <footer className="relative border-t border-border/80 bg-background/50 backdrop-blur-md">
      <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
        <div className="flex flex-col justify-between gap-10 md:flex-row">
          <div className="max-w-sm">
            <Link href="/" className="flex items-center gap-2.5">
              <Logo className="size-8" />
              <span className="font-mono text-sm font-bold">
                MatrixFlash<span className="text-primary text-glow">-Pro</span>
              </span>
            </Link>
            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
              {site.tagline[lang]}. {site.description[lang]}
            </p>
            <a
              href={site.github}
              target="_blank"
              rel="noreferrer"
              className="mt-5 inline-flex items-center gap-2 rounded-lg border border-border bg-secondary/50 px-3.5 py-2 text-sm text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
            >
              <GithubIcon className="size-4 text-primary" /> muhamedsahin/MatrixFlash_Pro
            </a>
          </div>

          <div className="grid grid-cols-2 gap-10 sm:grid-cols-3">
            <div>
              <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
                {t('DOKÜMANTASYON', 'DOCUMENTATION')}
              </h3>
              <ul className="mt-4 space-y-2.5">
                {topNav.slice(1).map((item) => (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      className="text-sm text-muted-foreground transition-colors hover:text-primary"
                    >
                      {item.label[lang]}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
                {t('MİMARİ & DONANIM', 'TECH STACK')}
              </h3>
              <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground font-mono">
                <li>NVIDIA {site.cuda}</li>
                <li>cuBLAS & cuSOLVER</li>
                <li>Tensor Core TF32</li>
                <li>{site.cpp} & CMake</li>
              </ul>
            </div>
            <div>
              <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
                {t('GELİŞTİRİCİ', 'DEVELOPER')}
              </h3>
              <ul className="mt-4 space-y-2.5 text-sm text-muted-foreground">
                <li className="font-medium text-foreground">{site.author}</li>
                <li>Bursa Teknik Üniv.</li>
                <li>{t('Bilgisayar Mühendisliği', 'Computer Engineering')}</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-border/60 pt-6 sm:flex-row">
          <p className="font-mono text-xs text-muted-foreground">
            © {new Date().getFullYear()} MatrixFlash-Pro · {site.author} · Open Source
          </p>
          <p className="font-mono text-xs text-primary/80">
            {t('GPU Hızında Yüksek Performanslı Matris Motoru', 'High-Performance GPU-Native Matrix Engine')}
          </p>
        </div>
      </div>
    </footer>
  )
}

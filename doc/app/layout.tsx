import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Space_Grotesk, JetBrains_Mono, Sora } from 'next/font/google'
import { LanguageProvider } from '@/lib/language-context'
import { IntroOverlay } from '@/components/intro-overlay'
import { ScrollProgressBar } from '@/components/scroll-progress-bar'
import { HudFrame } from '@/components/hud-frame'
import './globals.css'

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

const sora = Sora({
  subsets: ['latin', 'latin-ext'],
  variable: '--font-sora',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'MatrixFlash-Pro — CUDA-Accelerated C++17 Matrix & Tensor Library',
  description:
    'Shape-aware GEMM at cuBLAS-class speed, TF32 Tensor Cores, fused gemm+bias+act, Autograd, cuSOLVER, sparse CSR, async streams.',
  keywords: [
    'CUDA',
    'cuBLAS',
    'cuSOLVER',
    'Autograd',
    'Tensor',
    'Matrix',
    'GPU',
    'C++17',
    'MatrixFlash-Pro',
    'Deep Learning',
    'Linear Algebra',
  ],
  authors: [{ name: 'Muhammed Fatih Şahin' }],
}

export const viewport: Viewport = {
  colorScheme: 'dark',
  themeColor: '#0a1210',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="tr"
      className={`dark ${spaceGrotesk.variable} ${jetbrainsMono.variable} ${sora.variable}`}
    >
      <head>
      </head>
      <body className="font-sans antialiased selection:bg-primary/20 selection:text-primary">
        <LanguageProvider>
          <IntroOverlay />
          <ScrollProgressBar />
          <HudFrame />
          {/* Google Analytics */}
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-MNGNX6NSL7"
          strategy="afterInteractive"
        />

        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());

            gtag('config', 'G-MNGNX6NSL7');
          `}
        </Script>
          {children}
        </LanguageProvider>
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}

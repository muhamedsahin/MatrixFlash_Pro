import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Space_Grotesk, JetBrains_Mono, Sora } from 'next/font/google'
import Script from 'next/script'
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
  title: 'MatrixFlash-Pro — C++17 matrix library (AVX2 GEMM, cuBLAS)',
  description:
    'Host GEMM kernel avx2-4x16 on a Ryzen 7 5800H. Device GEMM is cuBLAS. CUDA 13.4 is optional. cuDNN is force-disabled. Apache-2.0.',
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

import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Space_Grotesk, JetBrains_Mono } from 'next/font/google'
import { LanguageProvider } from '@/lib/language-context'
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

export const metadata: Metadata = {
  title: 'MatrixFlash-Pro — CUDA-Accelerated C++17 Matrix & Tensor Library',
  description:
    'cuBLAS GEMM, TF32 Tensor Cores, Autograd tape engine, cuSOLVER linalg, sparse CSR, fused kernels, and async stream pool.',
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
  themeColor: '#0a0f0a',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="tr" className={`dark ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}>
      <body className="font-sans antialiased selection:bg-primary/20 selection:text-primary">
        <LanguageProvider>{children}</LanguageProvider>
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}

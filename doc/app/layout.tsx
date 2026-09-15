import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import { Space_Grotesk, JetBrains_Mono } from 'next/font/google'
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
  title: 'MatrixFlash-Pro — CUDA Tabanlı Yüksek Performanslı Matris Kütüphanesi',
  description:
    'Tiled CUDA matris çarpımı, GPU üzerinde çalışan dönüşümler ve aktivasyon fonksiyonları. Modern C++17 ile yazılmış, okunabilir ve hızlı matris işlem kütüphanesi.',
  generator: 'v0.app',
  keywords: [
    'CUDA',
    'CUDA kernels',
    'matris',
    'matrix',
    'GPU',
    'C++17',
    'MatrixFlash-Pro',
    'derin öğrenme',
    'doğrusal cebir',
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
      <body className="font-sans antialiased">
        {children}
        {process.env.NODE_ENV === 'production' && <Analytics />}
      </body>
    </html>
  )
}

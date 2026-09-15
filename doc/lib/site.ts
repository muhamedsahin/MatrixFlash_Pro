export const site = {
  name: 'MatrixFlash-Pro',
  tagline: 'CUDA Tabanlı, Yüksek Performanslı C++17 Matris Kütüphanesi',
  github: 'https://github.com/muhamedsahin/MatrixFlash_Pro',
  author: 'Muhammed Fatih Şahin',
  cpp: 'C++17',
  cuda: '12.3+',
}

export const topNav = [
  { label: 'Ana Sayfa', href: '/' },
  { label: 'Başlangıç', href: '/docs' },
  { label: 'Matris Nedir?', href: '/docs/matris-nedir' },
  { label: 'API', href: '/docs/api' },
  { label: 'Örnekler', href: '/docs/ornekler' },
]

export type DocLink = { label: string; href: string; badge?: string }
export type DocSection = { title: string; items: DocLink[] }

export const docsNav: DocSection[] = [
  {
    title: 'Başlarken',
    items: [
      { label: 'Genel Bakış', href: '/docs' },
      { label: 'Gereksinimler', href: '/docs#gereksinimler' },
      { label: 'Kurulum & Derleme', href: '/docs#kurulum' },
      { label: 'Hızlı Başlangıç', href: '/docs#hizli-baslangic' },
    ],
  },
  {
    title: 'Temeller',
    items: [
      { label: 'Matris Nedir?', href: '/docs/matris-nedir' },
      { label: 'GPU & cuBLAS Mantığı', href: '/docs/matris-nedir#gpu' },
      { label: 'Zincirleme API', href: '/docs/matris-nedir#zincirleme' },
    ],
  },
  {
    title: 'API Referansı',
    items: [
      { label: 'Oluşturucular', href: '/docs/api#olusturucular' },
      { label: 'Elementwise', href: '/docs/api#elementwise' },
      { label: 'Yayınlama (Broadcast)', href: '/docs/api#broadcast' },
      { label: 'İndirgemeler', href: '/docs/api#indirgemeler' },
      { label: 'Dönüşüm & Aktivasyon', href: '/docs/api#donusum' },
      { label: 'İleri Cebir', href: '/docs/api#ileri' },
      { label: 'Kalıcılık', href: '/docs/api#kalicilik' },
    ],
  },
  {
    title: 'Kaynaklar',
    items: [{ label: 'Örnekler', href: '/docs/ornekler' }],
  },
]

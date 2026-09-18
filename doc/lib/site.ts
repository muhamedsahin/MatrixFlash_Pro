export const site = {
  name: 'MatrixFlash-Pro',
  tagline: {
    tr: 'CUDA Tabanlı, Yüksek Performanslı C++17 Matris & Tensör Kütüphanesi',
    en: 'CUDA-Accelerated High-Performance C++17 Matrix & Tensor Library',
  },
  description: {
    tr: 'cuBLAS matris çarpımı, TF32 Tensor Core desteği, Autograd motoru, cuSOLVER doğrusal cebir, seyrek (sparse) matrisler ve asenkron stream havuzu.',
    en: 'cuBLAS GEMM, TF32 Tensor Core support, Autograd reverse-mode tape, cuSOLVER linalg, sparse CSR matrices, and asynchronous stream pool.',
  },
  github: 'https://github.com/muhamedsahin/MatrixFlash_Pro',
  author: 'Muhammed Fatih Şahin',
  cpp: 'C++17',
  cuda: 'CUDA 12.3+',
  version: '2.0.0-PRO',
}

export const topNav = [
  { label: { tr: 'Ana Sayfa', en: 'Home' }, href: '/' },
  { label: { tr: 'Başlangıç', en: 'Getting Started' }, href: '/docs' },
  { label: { tr: 'Matris Dersleri', en: 'Matrix Math Course' }, href: '/docs/matris-nedir' },
  { label: { tr: 'API Referansı', en: 'API Reference' }, href: '/docs/api' },
  { label: { tr: 'Örnekler', en: 'Examples' }, href: '/docs/ornekler' },
]

export type DocLink = {
  label: { tr: string; en: string }
  href: string
  badge?: string
}

export type DocSection = {
  title: { tr: string; en: string }
  items: DocLink[]
}

export const docsNav: DocSection[] = [
  {
    title: { tr: 'Başlarken', en: 'Getting Started' },
    items: [
      { label: { tr: 'Genel Bakış & Mimari', en: 'Overview & Architecture' }, href: '/docs' },
      { label: { tr: 'Gereksinimler & Kurulum', en: 'Prerequisites & Build' }, href: '/docs#kurulum' },
      { label: { tr: 'Hızlı Başlangıç (Quickstart)', en: 'Quickstart' }, href: '/docs#hizli-baslangic' },
      { label: { tr: 'Fonksiyon Çağrı Sırası & Yaşam Döngüsü', en: 'Calling Sequence & Lifecycle' }, href: '/docs#yasam-dongusu' },
      { label: { tr: 'Bellek Modu & Fail-Fast Güvenlik', en: 'Memory Modes & Fail-Fast Safety' }, href: '/docs#bellek-modu' },
      { label: { tr: 'İstisna & Hata Yönetimi', en: 'Error Hierarchy' }, href: '/docs#hata-yonetimi' },
    ],
  },
  {
    title: { tr: 'Matris Dersleri', en: 'Matrix Math Course' },
    items: [
      { label: { tr: '1. Matris Nedir?', en: '1. What is a Matrix?' }, href: '/docs/matris-nedir#bolum-1' },
      { label: { tr: '2. Temel Aritmetik & Transpoz', en: '2. Basic Arithmetic & Transpose' }, href: '/docs/matris-nedir#bolum-2' },
      { label: { tr: '3. Matris Çarpımı & Geometrik Sezgi', en: '3. Matmul & Geometric Intuition' }, href: '/docs/matris-nedir#bolum-3' },
      { label: { tr: '4. İleri Cebir (Det, Ters, SVD, Özdeğer)', en: '4. Advanced Linalg (Det, Inv, SVD, Eigen)' }, href: '/docs/matris-nedir#bolum-4' },
      { label: { tr: '5. Derin Öğrenmede Matrisler & Backprop', en: '5. Deep Learning Matrices & Backprop' }, href: '/docs/matris-nedir#bolum-5' },
      { label: { tr: '6. GPU Mimarisi & CUDA Paralelliği', en: '6. GPU Architecture & CUDA Tiling' }, href: '/docs/matris-nedir#bolum-6' },
      { label: { tr: 'İnteraktif Çarpım Simülatörü', en: 'Interactive Matmul Simulator' }, href: '/docs/matris-nedir#simulator', badge: 'Interactive' },
    ],
  },
  {
    title: { tr: 'API Referansı', en: 'API Reference' },
    items: [
      { label: { tr: 'Çekirdek (Matrix & Tensor)', en: 'Core (Matrix & Tensor)' }, href: '/docs/api#core' },
      { label: { tr: 'Fabrika Metotları (Factories)', en: 'Factories' }, href: '/docs/api#factories' },
      { label: { tr: 'Operasyonlar & Redüksiyonlar', en: 'Operations & Reductions' }, href: '/docs/api#ops' },
      { label: { tr: 'Yayınlama (Broadcasting)', en: 'Broadcasting' }, href: '/docs/api#broadcast' },
      { label: { tr: 'cuSOLVER İleri Cebir (SVD/QR/Eig)', en: 'cuSOLVER Linear Algebra' }, href: '/docs/api#cusolver' },
      { label: { tr: 'Aktivasyon & Fused / In-Place', en: 'Activation, Fused & In-Place' }, href: '/docs/api#nn-fused' },
      { label: { tr: '2D Konvolüsyon & Pooling', en: '2D Conv & Pooling' }, href: '/docs/api#conv' },
      { label: { tr: 'Autograd Ters-Mod Motoru', en: 'Autograd Tape Engine' }, href: '/docs/api#autograd', badge: 'Tape' },
      { label: { tr: 'Seyrek Matrisler (Sparse CSR)', en: 'Sparse CSR Matrices' }, href: '/docs/api#sparse' },
      { label: { tr: 'İndeksleme & Embedding', en: 'Indexing & Embedding' }, href: '/docs/api#indexing' },
      { label: { tr: 'Sıfır Kopyalama Görünümleri (View)', en: 'Zero-Copy Views' }, href: '/docs/api#view' },
      { label: { tr: 'Cihaz İçi RNG (Device RNG)', en: 'Device RNG' }, href: '/docs/api#rng' },
      { label: { tr: 'Stream Havuzu & Asenkron', en: 'Stream Pool & Async' }, href: '/docs/api#streams' },
    ],
  },
  {
    title: { tr: 'Pratik Örnekler', en: 'Practical Examples' },
    items: [
      { label: { tr: 'Temel GPU Kullanımı', en: 'Basic GPU Usage' }, href: '/docs/ornekler#basic' },
      { label: { tr: 'Yüksek Performans & Fused', en: 'High Performance & Fused' }, href: '/docs/ornekler#high-perf' },
      { label: { tr: 'Autograd ile MLP Eğitimi', en: 'MLP Training with Autograd' }, href: '/docs/ornekler#autograd-mlp' },
      { label: { tr: '2D CNN (Conv2D & MaxPool)', en: '2D CNN Forward & Backward' }, href: '/docs/ornekler#cnn' },
      { label: { tr: 'Seyrek Matris & Embedding', en: 'Sparse Matmul & Embedding' }, href: '/docs/ornekler#sparse' },
      { label: { tr: 'Zero-Copy MatrixView', en: 'Zero-Copy MatrixView' }, href: '/docs/ornekler#view' },
      { label: { tr: 'Çoklu Stream & Asenkron', en: 'Multi-Stream Async Pipeline' }, href: '/docs/ornekler#streams' },
    ],
  },
]

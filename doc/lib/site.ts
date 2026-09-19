export const site = {
  name: 'MatrixFlash-Pro',
  tagline: {
    tr: 'cuBLAS-sınıfı hızda C++17 GPU matris & tensör motoru',
    en: 'C++17 GPU matrix & tensor engine at cuBLAS-class speed',
  },
  description: {
    tr: 'Shape-aware GEMM, TF32 Tensor Core, fused gemm+bias+act, Autograd, cuSOLVER, seyrek CSR ve stream havuzu — NVIDIA GPU’da ölçülmüş performans.',
    en: 'Shape-aware GEMM, TF32 Tensor Cores, fused gemm+bias+act, Autograd, cuSOLVER, sparse CSR and a stream pool — measured on NVIDIA GPUs.',
  },
  github: 'https://github.com/muhamedsahin/MatrixFlash_Pro',
  author: 'Muhammed Fatih Şahin',
  cpp: 'C++17',
  cuda: 'CUDA 12.3+',
  version: '2.1.0-PRO',
}

export const topNav = [
  { label: { tr: 'Ana Sayfa', en: 'Home' }, href: '/' },
  { label: { tr: 'Başlangıç', en: 'Getting Started' }, href: '/docs' },
  { label: { tr: 'Matris Dersleri', en: 'Matrix Math Course' }, href: '/docs/matris-nedir' },
  { label: { tr: 'Performans', en: 'Benchmarks' }, href: '/docs/performans' },
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
      { label: { tr: 'Genel Bakış & Mimari', en: 'Overview & Architecture' }, href: '/docs#genel-bakis' },
      { label: { tr: 'Gereksinimler & Kurulum', en: 'Prerequisites & Build' }, href: '/docs#gereksinimler' },
      { label: { tr: 'Hızlı Başlangıç (Quickstart)', en: 'Quickstart' }, href: '/docs#hizli-baslangic' },
      { label: { tr: 'Fonksiyon Çağrı Sırası & Yaşam Döngüsü', en: 'Calling Sequence & Lifecycle' }, href: '/docs#yasam-dongusu' },
      { label: { tr: 'Bellek Modu & Fail-Fast Güvenlik', en: 'Memory Modes & Fail-Fast Safety' }, href: '/docs#bellek-modu' },
      { label: { tr: 'İstisna & Hata Yönetimi', en: 'Error Hierarchy' }, href: '/docs#hata-yonetimi' },
      { label: { tr: 'Derleme & Kurulum Adımları', en: 'Build & Install Steps' }, href: '/docs#kurulum' },
      { label: { tr: 'Performans İpuçları & Ayar Rehberi', en: 'Performance Tips & Tuning Guide' }, href: '/docs#performans-ipuclari' },
    ],
  },
  {
    title: { tr: 'Matris Dersleri', en: 'Matrix Math Course' },
    items: [
      { label: { tr: '0. Matris Nedir? (Sıfırdan)', en: '0. What Is a Matrix? (From Zero)' }, href: '/docs/matris-nedir#bolum-0', badge: 'yeni' },
      { label: { tr: '0b. Toplama & Skaler', en: '0b. Add & Scalar' }, href: '/docs/matris-nedir#bolum-0b' },
      { label: { tr: '0c. Matris × Vektör', en: '0c. Matrix × Vector' }, href: '/docs/matris-nedir#bolum-0c' },
      { label: { tr: '0d. Matris × Matris', en: '0d. Matrix × Matrix' }, href: '/docs/matris-nedir#bolum-0d' },
      { label: { tr: '1. Vektör Uzayları ve Matris Cebiri', en: '1. Vector Spaces & Matrix Algebra' }, href: '/docs/matris-nedir#bolum-1' },
      { label: { tr: '2. Matris Çarpımı & Bileşke Dönüşümler', en: '2. Matmul & Composite Transformations' }, href: '/docs/matris-nedir#bolum-2' },
      { label: { tr: '3. Determinant, Ters ve Lineer Sistemler', en: '3. Determinants, Inverses & Linear Systems' }, href: '/docs/matris-nedir#bolum-3' },
      { label: { tr: '4. Dört Temel Alt Uzay & Rank-Nullity', en: '4. The Four Fundamental Subspaces & Rank-Nullity' }, href: '/docs/matris-nedir#bolum-4' },
      { label: { tr: '5. Özdeğer, Özvektör & Spektral Ayrışım', en: '5. Eigenvalues, Eigenvectors & Spectral Decomposition' }, href: '/docs/matris-nedir#bolum-5' },
      { label: { tr: '6. SVD ve Düşük-Rank Yaklaşımı', en: '6. SVD & Low-Rank Approximation' }, href: '/docs/matris-nedir#bolum-6' },
      { label: { tr: '7. En Küçük Kareler & Normal Denklemler', en: '7. Least Squares & Normal Equations' }, href: '/docs/matris-nedir#bolum-7' },
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
      { label: { tr: 'Normalizasyon & Dropout', en: 'Normalization & Dropout' }, href: '/docs/api#ml' },
      { label: { tr: 'Cihaz Yönetimi, DType & CUDA Yardımcıları', en: 'Device Management, DType & CUDA Helpers' }, href: '/docs/api#device' },
      { label: { tr: 'Math & Operator Motoru', en: 'Math & Operator Engine' }, href: '/docs/api#math-engine', badge: 'v1.1' },
      { label: { tr: 'Gelişmiş Lineer Cebir (LU, TRSM)', en: 'Extended LinAlg (LU, TRSM)' }, href: '/docs/api#linalg-extended', badge: 'v1.1' },
      { label: { tr: 'Bellek Motoru & Unified Memory', en: 'Memory Engine & Unified' }, href: '/docs/api#memory-engine', badge: 'v1.1' },
      { label: { tr: 'Backend Motoru & Multi-GPU', en: 'Backend & Multi-GPU' }, href: '/docs/api#backend-engine', badge: 'v1.1' },
      { label: { tr: 'Yürütme Motoru (CUDA Graph)', en: 'Execution Engine (CUDA Graph)' }, href: '/docs/api#execution-engine', badge: 'v1.1' },
      { label: { tr: 'Serileştirme Motoru (Checkpoint)', en: 'Serialization (Checkpoint)' }, href: '/docs/api#serialization-engine', badge: 'v1.1' },
      { label: { tr: 'Seyrek Biçimler (COO, CSC, SpGEMM)', en: 'Sparse Formats (COO, CSC)' }, href: '/docs/api#sparse-engine', badge: 'v1.1' },
      { label: { tr: 'Gelişmiş Autograd (NoGrad)', en: 'Extended Autograd (NoGrad)' }, href: '/docs/api#autograd-extended', badge: 'v1.1' },
      { label: { tr: 'Genişletilmiş RNG Motoru', en: 'Extended RNG Engine' }, href: '/docs/api#rng-extended', badge: 'v1.1' },
    ],
  },
  {
    title: { tr: 'Performans Karşılaştırma', en: 'Benchmark Comparison' },
    items: [
      { label: { tr: 'Metodoloji', en: 'Methodology' }, href: '/docs/performans#metodoloji' },
      { label: { tr: 'GEMM vs cuBLAS', en: 'GEMM vs cuBLAS' }, href: '/docs/performans#gemm', badge: 'ölçüldü' },
      { label: { tr: 'Piyasa Motorları (6+)', en: 'Market Engines (6+)' }, href: '/docs/performans#rakipler', badge: 'yeni' },
      { label: { tr: 'Ölçülen GFLOPS Tablosu', en: 'Measured GFLOPS Table' }, href: '/docs/performans#olculen-motorlar' },
      { label: { tr: 'Fused GEMM+Bias+ReLU', en: 'Fused GEMM+Bias+ReLU' }, href: '/docs/performans#fused-gemm' },
      { label: { tr: 'Nasıl Tekrarlanır', en: 'How to Reproduce' }, href: '/docs/performans#nasil-tekrarlanir' },
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

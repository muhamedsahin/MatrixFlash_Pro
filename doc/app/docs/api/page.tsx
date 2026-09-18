'use client'

import React, { useState } from 'react'
import { ApiEntry } from '@/components/docs/api-entry'
import { DocSection, InlineCode, PageHeader, Callout } from '@/components/docs/doc-ui'
import { useLanguage } from '@/lib/language-context'
import { Search, Filter, Layers, Zap, Boxes, Cpu, GitBranch, Hash, Eye } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function ApiPage() {
  const { lang, t } = useLanguage()
  const [searchTerm, setSearchTerm] = useState('')
  const [activeCategory, setActiveCategory] = useState<string>('all')

  const categories = [
    { id: 'all', label: { tr: 'Tümü', en: 'All' } },
    { id: 'core', label: { tr: 'Çekirdek & Bellek', en: 'Core & Memory' } },
    { id: 'factories', label: { tr: 'Oluşturucular', en: 'Factories' } },
    { id: 'ops', label: { tr: 'GEMM & İndirgemeler', en: 'GEMM & Reductions' } },
    { id: 'cusolver', label: { tr: 'cuSOLVER Cebir', en: 'cuSOLVER Linalg' } },
    { id: 'nn', label: { tr: 'NN & Fused/In-Place', en: 'NN & Fused/In-Place' } },
    { id: 'conv', label: { tr: 'Conv2D & Tensör', en: 'Conv2D & Tensor' } },
    { id: 'autograd', label: { tr: 'Autograd Motoru', en: 'Autograd Tape' } },
    { id: 'sparse', label: { tr: 'Seyrek CSR', en: 'Sparse CSR' } },
    { id: 'view', label: { tr: 'View & Streams', en: 'View & Streams' } },
  ]

  return (
    <article className="space-y-6">
      <PageHeader
        eyebrow={t('REFERANS REHBERİ', 'REFERENCE GUIDE')}
        title={t('Eksiksiz API Referansı', 'Complete API Reference')}
        description={t(
          'MatrixFlash-Pro v2.0 modüler mimarisindeki tüm C++ sınıfları, fonksiyon imzaları, parametre açıklamaları ve çalışma prensipleri.',
          'Comprehensive reference for all C++ classes, function signatures, parameter descriptions, and runtime behaviors in MatrixFlash-Pro v2.0.',
        )}
      />

      <Callout type="info" title={t('Ad Alanı ve Başlıklar', 'Namespace & Header Notice')}>
        {t(
          'Tüm fonksiyonlar ve sınıflar matrix_pro ad alanı altındadır. Tek şemsiye başlık #include "matrix_pro/matrix_pro.hpp" ile tüm modüllere erişebilirsiniz.',
          'All types and routines reside under the matrix_pro namespace. Include the single umbrella header #include "matrix_pro/matrix_pro.hpp" to access the full API.',
        )}
      </Callout>

      {/* SEARCH AND FILTER BAR */}
      <div className="rounded-2xl border border-border/80 bg-card/60 p-4 shadow-sm backdrop-blur-md">
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder={t('Fonksiyon veya sınıf ara (örn: matmul, relu_, svd, Variable)...', 'Search function or class (e.g. matmul, relu_, svd, Variable)...')}
            className="w-full rounded-xl border border-border/80 bg-black/40 pl-10 pr-4 py-2 text-sm font-mono text-foreground placeholder:text-muted-foreground/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        {/* Category Pills */}
        <div className="mt-3 flex flex-wrap gap-1.5">
          {categories.map((cat) => (
            <button
              key={cat.id}
              type="button"
              onClick={() => setActiveCategory(cat.id)}
              className={cn(
                'rounded-lg px-2.5 py-1 text-xs font-mono font-medium transition-all',
                activeCategory === cat.id
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'bg-secondary/40 text-muted-foreground hover:bg-secondary hover:text-foreground',
              )}
            >
              {cat.label[lang]}
            </button>
          ))}
        </div>
      </div>

      {/* 1. CORE & MEMORY */}
      {(activeCategory === 'all' || activeCategory === 'core') && (
        <section className="space-y-5">
          <DocSection id="core" title={t('1. Çekirdek Sınıflar & Bellek Modeli', '1. Core Classes & Memory Model')}>
            <p>
              {t(
                'Matrix ve Tensor nesneleri; CPU RAM ve GPU VRAM arasındaki veri tutarlılığını fail-fast kurallarıyla yönetir.',
                'Matrix and Tensor classes manage synchronization between CPU and GPU memory with fail-fast safety.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="Matrix(rows, cols, MemoryMode)"
            signature="Matrix(std::size_t rows, std::size_t cols, MemoryMode mode = MemoryMode::host_and_device)"
            badge="constructor"
            params={[
              { name: 'rows', type: 'std::size_t', desc: t('Matris satır sayısı (M)', 'Row dimension (M)') },
              { name: 'cols', type: 'std::size_t', desc: t('Matris sütun sayısı (N)', 'Column dimension (N)') },
              { name: 'mode', type: 'MemoryMode', desc: t('host_and_device (varsayılan) veya device_only (host kopyası oluşturmaz)', 'host_and_device (default) or device_only (skips host RAM mirror)') },
            ]}
            returns="Matrix"
            example={`Matrix m(1024, 1024, MemoryMode::device_only); // 0 PCIe yükü`}
          >
            {t(
              'Belirtilen boyutlarda 2B GPU matrisi tahsis eder. device_only modu büyük modellerde RAM tüketimini ve gereksiz veri aktarımlarını önler.',
              'Allocates a 2D GPU matrix. device_only mode saves system RAM and bypasses PCIe traffic until download() is explicitly invoked.',
            )}
          </ApiEntry>

          <ApiEntry
            name="Matrix::download"
            signature="void download()"
            badge="synchronization"
            returns="void"
            example={`Matrix c = a * b; // GPU yazdı, host bayat
c.download();      // Host mirror GPU'dan güncellenir
float val = c.at(0, 0); // Güvenli okuma!`}
          >
            {t(
              'GPU device tamponundaki en güncel veriyi CPU host mirror tamponuna çeker. Fail-fast stale-mirror kuralı gereğince, GPU yazımı sonrasında download() çağrılmadan at() veya data() çağrılırsa kütüphane std::runtime_error fırlatır.',
              'Transfers updated device buffer contents back to the host mirror. Under the fail-fast contract, invoking at() or data() on stale host buffers throws immediately.',
            )}
          </ApiEntry>

          <ApiEntry
            name="Tensor(shape, MemoryMode)"
            signature="explicit Tensor(std::vector<std::size_t> shape, MemoryMode mode = MemoryMode::host_and_device)"
            badge="rank-N"
            params={[
              { name: 'shape', type: 'std::vector<size_t>', desc: t('Boyut vektörü (ör. {16, 3, 32, 32})', 'Shape dimensions vector (e.g. {16, 3, 32, 32})') },
            ]}
            returns="Tensor"
            example={`Tensor cnn_input({32, 3, 64, 64}, MemoryMode::device_only);`}
          >
            {t(
              'NCHW formatında çok boyutlu tensör kabı. 2B konvolüsyon ve havuzlama katmanları bu nesne üzerinde çalışır.',
              'Rank-N multi-dimensional tensor container. 2D convolution and pooling operations execute on Tensor instances.',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 2. FACTORIES */}
      {(activeCategory === 'all' || activeCategory === 'factories') && (
        <section className="space-y-5">
          <DocSection id="factories" title={t('2. Matris Oluşturucular (Factories)', '2. Matrix Factories')}>
            <p>
              {t(
                'GPU üzerinde doğrudan veri tahsis eden ve başlatan statik fabrika metotları.',
                'Static factory methods that allocate and initialize memory directly on the GPU.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="Matrix::randn_gpu"
            signature="static Matrix randn_gpu(std::size_t rows, std::size_t cols)"
            badge="device-rng"
            params={[
              { name: 'rows', type: 'std::size_t', desc: t('Satır sayısı', 'Rows') },
              { name: 'cols', type: 'std::size_t', desc: t('Sütun sayısı', 'Columns') },
            ]}
            returns="Matrix"
            example={`Matrix r = Matrix::randn_gpu(512, 512); // Doğrudan GPU üzerinde N(0,1)`}
          >
            {t(
              'Box-Muller dönüşümü ve MurmurHash3 sayaç tabanlı cihaz çekirdeği ile doğrudan GPU üzerinde standart normal dağılımlı matris üretir (0 host gidiş-dönüşü).',
              'Generates standard normally distributed elements directly on GPU via Box-Muller transformation and counter-based MurmurHash3 kernels.',
            )}
          </ApiEntry>

          <ApiEntry
            name="Matrix::identity"
            signature="static Matrix identity(std::size_t size)"
            badge="static"
            returns="Matrix"
            example={`Matrix I = Matrix::identity(4); // 4x4 birim matris`}
          >
            {t(
              'Köşegeni 1, diğer hücreleri 0 olan kare birim matris üretir.',
              'Constructs an identity matrix with ones along the main diagonal and zeros elsewhere.',
            )}
          </ApiEntry>

          <ApiEntry
            name="Matrix::zeros & ones"
            signature="static Matrix zeros(std::size_t rows, std::size_t cols)"
            badge="static"
            returns="Matrix"
            example={`Matrix z = Matrix::zeros(256, 256);
Matrix o = Matrix::ones(256, 256);`}
          >
            {t(
              'zeros doğrudan cudaMemsetAsync ile sıfırlanır; ones ise GPU üzerinde 1.0f değeriyle başlatılır.',
              'zeros invokes optimized cudaMemsetAsync; ones fills values on device.',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 3. GEMM & REDUCTIONS */}
      {(activeCategory === 'all' || activeCategory === 'ops') && (
        <section className="space-y-5">
          <DocSection id="ops" title={t('3. cuBLAS GEMM & GPU İndirgemeleri', '3. cuBLAS GEMM & GPU Reductions')}>
            <p>
              {t(
                'cuBLAS hızlandırmalı matris çarpımı ve GPU üzerinde tek geçişte çalışan istatistiksel indirgemeler.',
                'cuBLAS-accelerated matrix multiplication and single-pass GPU statistical reductions.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="operator* (cuBLAS GEMM)"
            signature="Matrix operator*(const Matrix& other) const"
            badge="cuBLAS + TF32"
            returns="Matrix"
            example={`Matrix c = a * b; // M x K ve K x N -> M x N`}
          >
            {t(
              'NVIDIA cuBLAS SGEMM ile donanım sınırında TFLOPS hızında matris çarpımı gerçekleştirir. Ampere/Ada/Hopper GPU’larda TF32 Tensor Core modu devrededir.',
              'Performs hardware-saturating GEMM via NVIDIA cuBLAS SGEMM, automatically activating TF32 Tensor Cores on supported architectures.',
            )}
          </ApiEntry>

          <ApiEntry
            name="sum, mean, argmax, argmin"
            signature="float sum() const; float mean() const; std::size_t argmax() const;"
            badge="reduction"
            returns="float / size_t"
            example={`float total = m.sum();
float avg = m.mean();
std::size_t best_idx = m.argmax();`}
          >
            {t(
              'GPU paylaşımlı bellek (shared memory) blok indirgemeleri ile tüm matrisi tek bir kernel başlatmasında indirger.',
              'Collapses the entire matrix in a single GPU kernel launch utilizing parallel tree reduction.',
            )}
          </ApiEntry>

          <ApiEntry
            name="broadcast_add & broadcast_multiply"
            signature="Matrix broadcast_add(const Matrix& vector) const"
            badge="broadcasting"
            returns="Matrix"
            example={`// x: (32 x 64), bias: (1 x 64)
Matrix out = x.broadcast_add(bias);`}
          >
            {t(
              'NumPy tarzı 2B yayınlama. Vektörü satır veya sütun boyunca tüm matris elemanlarına yayarak toplar veya çarpar.',
              'General NumPy-style 2D broadcasting along row or column dimensions.',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 4. cuSOLVER LINALG */}
      {(activeCategory === 'all' || activeCategory === 'cusolver') && (
        <section className="space-y-5">
          <DocSection id="cusolver" title={t('4. cuSOLVER İleri Doğrusal Cebir', '4. cuSOLVER Linear Algebra')}>
            <p>
              {t(
                'NVIDIA cuSOLVER kütüphanesi tarafından hızlandırılan matris ayrışımları ve çözücüler.',
                'Decompositions and linear solvers accelerated by NVIDIA cuSOLVER.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="Matrix::svd"
            signature="SVDResult svd() const"
            badge="cuSOLVER"
            returns="SVDResult { Matrix u; Matrix s; Matrix vt; }"
            example={`SVDResult res = a.svd();
// res.u: Sol tekil vektörler
// res.s: Tekil değerler (1 x min(M,N))
// res.vt: Sağ tekil vektörlerin devriği`}
          >
            {t(
              'A = U * Σ * V^T tekil değer ayrışımını GPU üzerinde hesaplar. PCA ve boyut indirgeme için temel taş.',
              'Computes Singular Value Decomposition A = U * Σ * V^T directly on the GPU.',
            )}
          </ApiEntry>

          <ApiEntry
            name="Matrix::cholesky"
            signature="Matrix cholesky() const"
            badge="cuSOLVER"
            returns="Matrix (Alt üçgen L matrisi)"
            example={`Matrix l = cov_matrix.cholesky(); // A = L * L^T`}
          >
            {t(
              'Simetrik ve pozitif tanımlı matrisler için Cholesky çarpanlarına ayırma (A = L * L^T).',
              'Computes Cholesky factorization (A = L * L^T) for symmetric positive-definite matrices.',
            )}
          </ApiEntry>

          <ApiEntry
            name="Matrix::solve & pinv"
            signature="Matrix solve(const Matrix& rhs) const; Matrix pinv() const;"
            badge="cuSOLVER"
            returns="Matrix"
            example={`Matrix x = a.solve(b); // Ax = b
Matrix a_pinv = a.pinv(); // Moore-Penrose tersi`}
          >
            {t(
              'Lineer denklem sistemlerini LU/QR ile çözer veya tekil olmayan/dikdörtgen matrisler için Moore-Penrose sözde tersini (pinv) üretir.',
              'Solves linear systems Ax = b or computes Moore-Penrose pseudo-inverse (pinv).',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 5. NN & FUSED/INPLACE */}
      {(activeCategory === 'all' || activeCategory === 'nn') && (
        <section className="space-y-5">
          <DocSection id="nn-fused" title={t('5. NN Aktivasyonları, Fused & In-Place', '5. NN Activations, Fused & In-Place')}>
            <p>
              {t(
                'GPU bellek bant genişliğini en üst düzeye çıkaran in-place operasyonlar ve birleştirilmiş (fused) kernel zincirleri.',
                'In-place operators and fused chains optimizing memory bandwidth in neural networks.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="fused_bias_gelu & fused_sigmoid_mul"
            signature="Matrix fused_bias_gelu(const Matrix& x, const Matrix& bias)"
            badge="fused-chain"
            returns="Matrix"
            example={`Matrix y = fused_bias_gelu(x, bias); // Tek kernelda x + bias ve GeLU`}
          >
            {t(
              'İki veya daha fazla GPU küresel bellek gidiş-dönüşünü tek bir çekirdekte birleştirir. Transformer MLP ve LSTM katmanlarında %40+ hız artışı sağlar.',
              'Collapses multiple global memory transactions into a single kernel pass, boosting throughput in Transformer and LSTM blocks.',
            )}
          </ApiEntry>

          <ApiEntry
            name="relu_, sigmoid_, gelu_, add_"
            signature="Matrix& relu_(Matrix& self); Matrix& add_(Matrix& self, float val);"
            badge="in-place"
            returns="Matrix&"
            example={`relu_(activated); // Giriş tamponunu doğrudan günceller (0 VRAM tahsisi)`}
          >
            {t(
              'Sonucu yeni bir matrise yazmak yerine doğrudan giriş matrisinin üzerine yazar; bellek ayırma maliyetini sıfırlar.',
              'Writes output directly into input buffer, saving an allocation and a memory pass per invocation.',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 6. CONV2D & TENSOR */}
      {(activeCategory === 'all' || activeCategory === 'conv') && (
        <section className="space-y-5">
          <DocSection id="conv" title={t('6. 2D Konvolüsyon ve Havuzlama (CNN)', '6. 2D Convolution & Pooling (CNN')}>
            <p>
              {t(
                'NCHW formatında ileri ve geri yayılım destekli konvolüsyonel sinir ağı katmanları.',
                'Forward and backward differentiable CNN primitives in NCHW format.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="conv2d"
            signature="Tensor conv2d(const Tensor& input, const Tensor& weights, const Tensor& bias, size_t stride = 1, size_t padding = 0)"
            badge="cnn"
            returns="Tensor"
            example={`Tensor out = conv2d(x, w, b, /*stride=*/1, /*padding=*/1);`}
          >
            {t(
              'Girdi: (N, C, H, W), Ağırlık: (K, C, kH, kW), Bias: (K). 2B çapraz korelasyon konvolüsyonu GPU üzerinde hesaplar.',
              'Executes 2D cross-correlation convolution over rank-4 tensors on CUDA.',
            )}
          </ApiEntry>

          <ApiEntry
            name="max_pool2d & avg_pool2d"
            signature="Tensor max_pool2d(const Tensor& input, size_t kernel_size, size_t stride = 1, size_t padding = 0)"
            badge="pooling"
            returns="Tensor"
            example={`Tensor pooled = max_pool2d(out, 2, 2);`}
          >
            {t(
              'Uzamsal boyutları kernel boyutuna göre küçülten maksimum ve ortalama havuzlama katmanları.',
              'Downsamples spatial dimensions via max or average pooling operations.',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 7. AUTOGRAD TAPE */}
      {(activeCategory === 'all' || activeCategory === 'autograd') && (
        <section className="space-y-5">
          <DocSection id="autograd" title={t('7. Autograd Ters-Mod Otomatik Türev', '7. Autograd Reverse-Mode Engine')}>
            <p>
              {t(
                'PyTorch tarzı hesaplama bandı. Variable ve VarTensor nesneleri ile dinamik geri yayılım.',
                'PyTorch-style computational graph tape for dynamic reverse-mode automatic differentiation.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="Variable::backward"
            signature="void backward()"
            badge="autograd"
            returns="void"
            example={`Variable loss = mse_loss(y, target);
loss.backward();
const Matrix& grad_w = w.grad();`}
          >
            {t(
              'Kayıp değerinden başlayarak hesaplama grafiğindeki tüm requires_grad=true yapraklara zincir kuralı gradyanlarını biriktirir.',
              'Traverses the computation tape in topological order, propagating gradients to leaf nodes.',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 8. SPARSE CSR */}
      {(activeCategory === 'all' || activeCategory === 'sparse') && (
        <section className="space-y-5">
          <DocSection id="sparse" title={t('8. Seyrek Matrisler (Sparse CSR & SpMV)', '8. Sparse CSR Matrices & SpMV')}>
            <p>
              {t(
                'Graf sinir ağları ve embedding tabloları için sıkıştırılmış satır (CSR) seyrek matris formatı.',
                'Compressed Sparse Row (CSR) format optimized for graph models and sparse-dense matrix multiplication.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="SparseCSR::from_dense & spmv"
            signature="static SparseCSR from_dense(const Matrix& dense, float threshold = 0.0f); Matrix spmv(const SparseCSR& a, const Matrix& x);"
            badge="sparse"
            returns="SparseCSR / Matrix"
            example={`SparseCSR sp = SparseCSR::from_dense(dense, 1e-4f);
Matrix y = spmv(sp, x); // y = A * x`}
          >
            {t(
              'Yoğun matrisi CSR formatına sıkıştırır; spmv ile seyrek matris-vektör çarpımını GPU üzerinde yürütür.',
              'Compresses dense matrix to CSR buffers; spmv executes sparse matrix-vector multiplication on CUDA.',
            )}
          </ApiEntry>
        </section>
      )}

      {/* 9. VIEW & STREAMS */}
      {(activeCategory === 'all' || activeCategory === 'view') && (
        <section className="space-y-5">
          <DocSection id="view" title={t('9. Sıfır Kopyalama Görünümleri & Stream Havuzu', '9. Zero-Copy Views & Stream Pool')}>
            <p>
              {t(
                'Adım (stride) tabanlı sıfır kopyalama MatrixView ve asenkron çoklu stream eş zamanlılığı.',
                'Stride-based zero-copy MatrixView and asynchronous multi-stream execution pool.',
              )}
            </p>
          </DocSection>

          <ApiEntry
            name="transpose_view & slice_view"
            signature="MatrixView transpose_view(Matrix& matrix); MatrixView slice_view(Matrix& matrix, size_t r0, size_t r1, size_t c0, size_t c1);"
            badge="zero-copy"
            returns="MatrixView"
            example={`MatrixView v = transpose_view(m); // 0 bayt VRAM ayrılır!`}
          >
            {t(
              'Adımları manipüle ederek bellekte yeni bir tahsis yapmadan anında transpoz veya dilim oluşturur.',
              'Manipulates strides to construct instant transposed or sliced views without copying VRAM.',
            )}
          </ApiEntry>

          <ApiEntry
            name="pool_stream & argmax_async"
            signature="cudaStream_t pool_stream(int slot); void argmax_async(const Matrix& m, size_t* out, callback);"
            badge="streams"
            returns="cudaStream_t / void"
            example={`cudaStream_t s = pool_stream(1);
argmax_async(m, pinned_slot, [](size_t best) { ... });`}
          >
            {t(
              'kStreamPoolSize = 4 ile eş zamanlı CUDA akışları sağlar. argmax_async CPU iş parçacığını bloklamadan pinned belleğe sonuç yazar.',
              'Provides concurrent CUDA stream pool and non-blocking reductions posting to pinned host memory.',
            )}
          </ApiEntry>
        </section>
      )}
    </article>
  )
}

'use client'

import React from 'react'
import Link from 'next/link'
import {
  Cpu,
  Layers,
  Zap,
  ArrowRight,
  BookOpen,
  Sparkles,
  Compass,
  Maximize2,
  Activity,
  CheckCircle2,
} from 'lucide-react'
import { CodeBlock } from '@/components/code-block'
import { Math } from '@/components/math'
import { MatrixFigure } from '@/components/docs/matrix-figure'
import {
  Callout,
  DocSection,
  InlineCode,
  PageHeader,
  SubHeading,
} from '@/components/docs/doc-ui'
import { MatrixCalculator } from '@/components/interactive/matrix-calculator'
import { useLanguage } from '@/lib/language-context'

export default function MatrisDersleriPage() {
  const { lang, t } = useLanguage()

  return (
    <article className="space-y-6">
      <PageHeader
        eyebrow={t('MATEMATİKSEL TEMELLER & PEDAGOJİK KURS', 'MATHEMATICAL FOUNDATIONS & COURSE')}
        title={t(
          'Matematiksel Olarak Matris Nedir? — Sıfırdan Derin Öğrenmeye',
          'What is a Matrix? — From Foundations to Deep Learning',
        )}
        description={t(
          'Matrislerin sezgisel tanımı, geometrik dönüşüm anlamı, temel ve ileri cebir kuralları, yapay sinir ağlarındaki rolü ve GPU mimarisinin neden matrisler için yaratıldığına dair eksiksiz rehber.',
          'An intuitive, mathematically rigorous guide to matrices: geometric transformation intuition, arithmetic, advanced decompositions, neural network forward/backward mechanics, and GPU architecture.',
        )}
      />

      {/* İÇİNDEKİLER QUICK BAR */}
      <div className="rounded-xl border border-border/80 bg-card/40 p-4 backdrop-blur-md">
        <span className="font-mono text-xs uppercase tracking-wider text-primary font-bold">
          {t('DERS PLANI (BÖLÜMLER)', 'COURSE SYLLABUS')}
        </span>
        <div className="mt-2.5 grid grid-cols-1 gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3">
          <a href="#bolum-1" className="text-muted-foreground hover:text-primary transition-colors">
            1. {t('Matris Nedir & Tensör Hiyerarşisi', 'What is a Matrix & Tensor Hierarchy')}
          </a>
          <a href="#bolum-2" className="text-muted-foreground hover:text-primary transition-colors">
            2. {t('Temel İşlemler & Transpoz', 'Basic Arithmetic & Transpose')}
          </a>
          <a href="#bolum-3" className="text-muted-foreground hover:text-primary transition-colors">
            3. {t('Matris Çarpımı & Geometrik Sezgi', 'Matrix Multiplication & Geometry')}
          </a>
          <a href="#simulator" className="text-primary font-semibold hover:underline">
            ★ {t('İnteraktif Çarpım Simülatörü', 'Interactive Matmul Simulator')}
          </a>
          <a href="#bolum-4" className="text-muted-foreground hover:text-primary transition-colors">
            4. {t('İleri Cebir (Det, Ters, Özdeğer, SVD)', 'Advanced Linalg (Det, Inv, Eigen, SVD)')}
          </a>
          <a href="#bolum-5" className="text-muted-foreground hover:text-primary transition-colors">
            5. {t('Yapay Zekada Matrisler & Backprop', 'Matrices in Deep Learning & Backprop')}
          </a>
          <a href="#bolum-6" className="text-muted-foreground hover:text-primary transition-colors">
            6. {t('GPU Mimarisi & CUDA Paralelliği', 'GPU Architecture & CUDA Parallelism')}
          </a>
        </div>
      </div>

      {/* BÖLÜM 1 */}
      <DocSection id="bolum-1" title={t('Bölüm 1: Matris Nedir? — Sayılardan Boyutlara', 'Chapter 1: What is a Matrix? — From Numbers to Dimensions')}>
        <p>
          {lang === 'tr' ? (
            <>
              Matematikte bir <strong className="text-foreground">matris</strong>; sayıların veya sembollerin
              dikdörtgen bir ızgara (satır ve sütunlar) biçiminde düzenlenmiş halidir. Ancak modern bilgisayar
              biliminde ve yapay zekada matris sadece bir veri tablosu değil, uzaydaki vektörleri dönüştüren
              <strong className="text-primary"> geometrik bir operatördür</strong>.
            </>
          ) : (
            <>
              In mathematics, a <strong className="text-foreground">matrix</strong> is a rectangular array of
              numbers arranged in rows and columns. In modern computing and machine learning, however, a matrix is
              not merely a data table — it is a <strong className="text-primary">geometric linear transformation</strong>.
            </>
          )}
        </p>

        <Math display>
          {'A = \\begin{bmatrix} a_{11} & a_{12} & \\cdots & a_{1n} \\\\ a_{21} & a_{22} & \\cdots & a_{2n} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ a_{m1} & a_{m2} & \\cdots & a_{mn} \\end{bmatrix} \\in \\mathbb{R}^{m \\times n}'}
        </Math>

        <p>
          {lang === 'tr' ? (
            <>
              Burada <Math>{'m'}</Math> satır sayısını (satırlar yataydır), <Math>{'n'}</Math> ise sütun sayısını (sütunlar dikeydir)
              temsil eder. <Math>{'a_{ij}'}</Math> sembolü, <Math>{'i'}</Math>. satır ve <Math>{'j'}</Math>. sütunun kesişimindeki skaler değerdir.
            </>
          ) : (
            <>
              Here <Math>{'m'}</Math> denotes the number of rows (horizontal) and <Math>{'n'}</Math> denotes the number of columns (vertical).
              The entry <Math>{'a_{ij}'}</Math> sits at the intersection of the <Math>{'i'}</Math>-th row and <Math>{'j'}</Math>-th column.
            </>
          )}
        </p>

        <SubHeading>{t('Tensör Boyut Hiyerarşisi (Rank)', 'The Tensor Rank Hierarchy')}</SubHeading>
        <p>
          {t(
            'Yapay zeka ve doğrusal cebirde her veri yapısı aslında birer tensördür:',
            'In deep learning, every data structure is fundamentally a tensor categorized by its rank (dimensionality):',
          )}
        </p>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-4 my-4">
          <div className="rounded-xl border border-border/70 bg-card/40 p-4">
            <span className="font-mono text-xs text-primary font-bold">{t('Rank 0: Skaler', 'Rank 0: Scalar')}</span>
            <p className="mt-1 font-mono text-sm text-foreground">x = 3.14</p>
            <p className="mt-1 text-[11px] text-muted-foreground">{t('Tek bir sayı, boyutu yok ()', 'Single number, shape ()')}</p>
          </div>
          <div className="rounded-xl border border-border/70 bg-card/40 p-4">
            <span className="font-mono text-xs text-accent font-bold">{t('Rank 1: Vektör', 'Rank 1: Vector')}</span>
            <p className="mt-1 font-mono text-sm text-foreground">v = [x, y, z]</p>
            <p className="mt-1 text-[11px] text-muted-foreground">{t('1B dizi, boyut (N,)', '1D array, shape (N,)')}</p>
          </div>
          <div className="rounded-xl border border-primary/40 bg-primary/10 p-4 shadow-[0_0_15px_rgba(110,231,183,0.1)]">
            <span className="font-mono text-xs text-emerald-400 font-bold">{t('Rank 2: Matris', 'Rank 2: Matrix')}</span>
            <p className="mt-1 font-mono text-sm text-foreground">M = [M × N]</p>
            <p className="mt-1 text-[11px] text-muted-foreground">{t('2B tablo, Matrix(M, N)', '2D table, Matrix(M, N)')}</p>
          </div>
          <div className="rounded-xl border border-border/70 bg-card/40 p-4">
            <span className="font-mono text-xs text-amber-400 font-bold">{t('Rank N: Tensör', 'Rank N: Tensor')}</span>
            <p className="mt-1 font-mono text-sm text-foreground">T = [B, C, H, W]</p>
            <p className="mt-1 text-[11px] text-muted-foreground">{t('Çok boyutlu, Tensor(shape)', 'High-D array, Tensor(shape)')}</p>
          </div>
        </div>

        <MatrixFigure
          data={[
            ['a₁₁', 'a₁₂', 'a₁₃'],
            ['a₂₁', 'a₂₂', 'a₂₃'],
          ]}
          highlight={[[1, 1]]}
          caption={t('a₂₂ elemanı → 2. satır, 2. sütun', 'Entry a₂₂ → Row 2, Column 2')}
        />
      </DocSection>

      {/* BÖLÜM 2 */}
      <DocSection id="bolum-2" title={t('Bölüm 2: Temel Aritmetik & Transpoz', 'Chapter 2: Basic Arithmetic & Transpose')}>
        <SubHeading>{t('1. Eleman Bazında Toplama ve Çıkarma', '1. Elementwise Addition & Subtraction')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              İki matrisin toplanabilmesi veya çıkarılabilmesi için <strong className="text-foreground">aynı boyutlara</strong>{' '}
              sahip olması şarttır (<Math>{'A, B \\in \\mathbb{R}^{m \\times n}'}</Math>). İşlem karşılıklı elemanlar arasında yapılır:
            </>
          ) : (
            <>
              Two matrices can only be added or subtracted if they share the <strong className="text-foreground">exact same dimensions</strong>{' '}
              (<Math>{'A, B \\in \\mathbb{R}^{m \\times n}'}</Math>). The operation executes component-wise:
            </>
          )}
        </p>

        <Math display>{'(A + B)_{ij} = A_{ij} + B_{ij}'}</Math>

        <div className="flex flex-wrap items-center justify-center gap-3 my-4">
          <MatrixFigure data={[['2', '4'], ['1', '3']]} />
          <span className="font-mono text-2xl text-primary">+</span>
          <MatrixFigure data={[['5', '1'], ['2', '6']]} accent="accent" />
          <span className="font-mono text-2xl text-primary">=</span>
          <MatrixFigure data={[['7', '5'], ['3', '9']]} accent="chart-3" />
        </div>

        <SubHeading>{t('2. Skaler ile Çarpma', '2. Scalar Multiplication')}</SubHeading>
        <p>
          {t(
            'Bir matrisin bir skaler c sayısı ile çarpılması, matrisin her bir elemanının c ile çarpılması demektir:',
            'Multiplying a matrix by a scalar c scales every single component by c:',
          )}
        </p>
        <Math display>{'(c \\cdot A)_{ij} = c \\cdot A_{ij}'}</Math>

        <SubHeading>{t('3. Devrik (Transpose)', '3. The Transpose Operation')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              Bir matrisin devriği (<Math>{'A^T'}</Math>), satırları ile sütunlarının yer değiştirmesidir.{' '}
              <Math>{'m \\times n'}</Math> boyutundaki matris, <Math>{'n \\times m'}</Math> boyutuna dönüşür:
            </>
          ) : (
            <>
              The transpose of a matrix (<Math>{'A^T'}</Math>) swaps its rows and columns.{' '}
              An <Math>{'m \\times n'}</Math> matrix becomes an <Math>{'n \\times m'}</Math> matrix:
            </>
          )}
        </p>
        <Math display>{'(A^T)_{ij} = A_{ji}'}</Math>

        <Callout type="info" title={t('MatrixFlash-Pro Sıfır Kopyalama Transpoz', 'Zero-Copy Transpose in MatrixFlash-Pro')}>
          {t(
            'Geleneksel kütüphaneler transpoz alırken bellekte yeni bir kopya oluşturur. MatrixFlash-Pro view modülü (transpose_view), yalnızca satır ve sütun adımlarını (strides) takas ederek 0 bayt VRAM harcayarak anında transpoz oluşturabilir!',
            'Traditional libraries allocate new VRAM when transposing. MatrixFlash-Pro’s transpose_view swaps strides in O(1) time without allocating or copying a single byte in device memory!',
          )}
        </Callout>
      </DocSection>

      {/* BÖLÜM 3 */}
      <DocSection id="bolum-3" title={t('Bölüm 3: Matris Çarpımı & Geometrik Sezgi', 'Chapter 3: Matrix Multiplication & Geometric Intuition')}>
        <p>
          {lang === 'tr' ? (
            <>
              Matris çarpımı, doğrusal cebirin en temel, en güçlü ve hesaplama açısından en yoğun operasyonudur.
              İki matrisin çarpılabilmesi için birinci matrisin sütun sayısı ile ikinci matrisin satır sayısının{' '}
              <strong className="text-foreground">tamamen eşit olması gerekir</strong>:
            </>
          ) : (
            <>
              Matrix multiplication (GEMM) is the bedrock of linear algebra and the primary computational load
              in artificial intelligence. For multiplication to be valid, the inner dimensions must match:
            </>
          )}
        </p>

        <Math display>
          {'A \\in \\mathbb{R}^{m \\times k}, \\quad B \\in \\mathbb{R}^{k \\times n} \\implies C = A \\cdot B \\in \\mathbb{R}^{m \\times n}'}
        </Math>

        <p>
          {lang === 'tr' ? (
            <>
              Sonuç matrisinin her bir <Math>{'C_{ij}'}</Math> elemanı; <Math>{'A'}</Math> matrisinin <Math>{'i'}</Math>. satırı ile{' '}
              <Math>{'B'}</Math> matrisinin <Math>{'j'}</Math>. sütununun <strong className="text-primary">nokta çarpımıdır (dot product)</strong>:
            </>
          ) : (
            <>
              Each scalar entry <Math>{'C_{ij}'}</Math> in the resulting matrix is the{' '}
              <strong className="text-primary">inner dot product</strong> of the <Math>{'i'}</Math>-th row of <Math>{'A'}</Math>{' '}
              with the <Math>{'j'}</Math>-th column of <Math>{'B'}</Math>:
            </>
          )}
        </p>

        <Math display>
          {'C_{ij} = \\sum_{p=1}^{k} A_{ip} \\cdot B_{pj} = A_{i1}B_{1j} + A_{i2}B_{2j} + \\dots + A_{ik}B_{kj}'}
        </Math>

        <SubHeading>{t('Geometrik Anlam: Matris bir Lineer Dönüşümdür!', 'Geometric Intuition: Matrices are Transformations!')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              Bir vektörü <Math>{'x'}</Math> bir matris <Math>{'A'}</Math> ile çarptığınızda (<Math>{'Ax'}</Math>), aslında uzayı
              dönüştürürsünüz. Matrisin sütunları, standart eksen birim vektörlerinin (<Math>{'\\hat{i}'}</Math> ve <Math>{'\\hat{j}'}</Math>)
              dönüşüm sonrasında nereye düştüğünü söyler! Matrisler uzayı{' '}
              <strong className="text-foreground">döndürebilir, ölçekleyebilir (uzatıp/kısaltabilir) veya eğitebilir (shear)</strong>.
            </>
          ) : (
            <>
              When you multiply a vector <Math>{'x'}</Math> by a matrix <Math>{'A'}</Math> (<Math>{'Ax'}</Math>), you are performing
              a linear transformation on space. The columns of <Math>{'A'}</Math> describe where the basis vectors (<Math>{'\\hat{i}'}</Math>, <Math>{'\\hat{j}'}</Math>)
              land! A matrix can <strong className="text-foreground">rotate, scale, reflect, or shear</strong> space.
            </>
          )}
        </p>
      </DocSection>

      {/* İNTERAKTİF SİMÜLATÖR */}
      <section id="simulator" className="scroll-mt-24 my-8">
        <MatrixCalculator />
      </section>

      {/* BÖLÜM 4 */}
      <DocSection id="bolum-4" title={t('Bölüm 4: İleri Doğrusal Cebir (Determinant, Ters, SVD, Özdeğer)', 'Chapter 4: Advanced Linear Algebra (Determinant, Inverse, SVD, Eigen)')}>
        <SubHeading>{t('1. Determinant — Hacim Değişim Oranı', '1. The Determinant — Area/Volume Scaling Factor')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              Determinant (<Math>{'\\det(A)'}</Math>), bir kare matrisin uzayı dönüştürürken birim alan veya hacmi kaç katına
              çıkardığını gösteren skaler bir sayıdır. Örneğin bir <Math>{'2 \\times 2'}</Math> dönüşümün determinantı 3 ise,
              dönüşüm sonrasında tüm alanlar 3 kat büyür.
            </>
          ) : (
            <>
              The determinant (<Math>{'\\det(A)'}</Math>) measures how much a linear transformation scales areas or volumes.
              If a 2D transformation has <Math>{'\\det(A) = 3'}</Math>, every area in the plane triples after transformation.
            </>
          )}
        </p>
        <Math display>
          {'\\det \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix} = ad - bc'}
        </Math>
        <Callout type="warn" title={t('Tekil (Singular) Matris Uyarısı', 'Singular Matrix Alert')}>
          {t(
            'Eğer det(A) = 0 ise, matris uzayı bir alt boyuta ezer (örneğin 2B alanı 1B çizgiye indirir). Bilgi kaybolduğu için bu matrisin TERSİ ALINAMAZ (tekildir). MatrixFlash-Pro bu durumda SolverError fırlatır.',
            'If det(A) = 0, the transformation squashes space into a lower dimension (e.g. 2D plane onto a line). Information is lost, meaning the matrix is singular and NON-INVERTIBLE. MatrixFlash-Pro throws SolverError in this scenario.',
          )}
        </Callout>

        <SubHeading>{t('2. Ters Matris (Matrix Inverse)', '2. Matrix Inverse')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              Bir matrisin tersi (<Math>{'A^{-1}'}</Math>), o matrisin yaptığı dönüşümü tamamen geri alan (undo) matristir.
              Bir matris kendi tersi ile çarpıldığında birim matrisi (<Math>{'I'}</Math>) verir:
            </>
          ) : (
            <>
              The inverse of a matrix (<Math>{'A^{-1}'}</Math>) is the transformation that reverses or undoes what <Math>{'A'}</Math> did.
              Multiplying a matrix by its inverse yields the identity matrix (<Math>{'I'}</Math>):
            </>
          )}
        </p>
        <Math display>{'A \\cdot A^{-1} = A^{-1} \\cdot A = I'}</Math>

        <SubHeading>{t('3. Özdeğerler ve Özvektörler (Eigenvalues & Eigenvectors)', '3. Eigenvalues & Eigenvectors')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              Uzay dönüştürülürken çoğu vektörün yönü değişir. Ancak bazı özel vektörler vardır ki, yönleri kesinlikle değişmez,
              yalnızca bir <Math>{'\\lambda'}</Math> katsayısıyla uzar veya kısalır. Bu vektörlere{' '}
              <strong className="text-primary">özvektör (eigenvector)</strong>, o katsayıya ise{' '}
              <strong className="text-accent">özdeğer (eigenvalue)</strong> denir:
            </>
          ) : (
            <>
              During linear transformation, most vectors get knocked off their span. However, special vectors remain on the
              exact same line, merely scaling by a scalar factor <Math>{'\\lambda'}</Math>. These are{' '}
              <strong className="text-primary">eigenvectors</strong>, and <Math>{'\\lambda'}</Math> is the corresponding{' '}
              <strong className="text-accent">eigenvalue</strong>:
            </>
          )}
        </p>
        <Math display>{'A \\cdot v = \\lambda \\cdot v'}</Math>

        <SubHeading>{t('4. Tekil Değer Ayrışımı (SVD - Singular Value Decomposition)', '4. Singular Value Decomposition (SVD)')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              SVD; kare olmak zorunda olmayan herhangi bir <Math>{'m \\times n'}</Math> matrisi üç geometrik operasyonun
              bileşkesine ayırır: <strong className="text-foreground">Döndürme (U) × Ölçekleme (Σ) × Döndürme (Vᵀ)</strong>.
              PCA boyut indirgeme ve veri sıkıştırmada vazgeçilmezdir. MatrixFlash-Pro bunu doğrudan cuSOLVER üzerinden GPU ile çözer:
            </>
          ) : (
            <>
              SVD factors any arbitrary <Math>{'m \\times n'}</Math> matrix into three fundamental geometric stages:{' '}
              <strong className="text-foreground">Rotation (U) × Scaling (Σ) × Rotation (Vᵀ)</strong>.
              It is the backbone of dimensionality reduction (PCA) and latent factor models. MatrixFlash-Pro solves SVD natively via cuSOLVER:
            </>
          )}
        </p>
        <Math display>{'A = U \\cdot \\Sigma \\cdot V^T'}</Math>
      </DocSection>

      {/* BÖLÜM 5 */}
      <DocSection id="bolum-5" title={t('Bölüm 5: Yapay Zekada Matrisler — Neden Her Şey Matristir?', 'Chapter 5: Matrices in Deep Learning — Why Everything is a Matrix')}>
        <p>
          {lang === 'tr' ? (
            <>
              Modern yapay sinir ağları özünde devasa matris işlemlerinden ibarettir. Tam bağlı bir yapay sinir ağı katmanı
              (Dense / Linear Layer) matematiksel olarak şu formülle ifade edilir:
            </>
          ) : (
            <>
              Modern artificial neural networks are essentially cascades of high-dimensional matrix multiplications.
              A fully-connected dense layer is formulated mathematically as:
            </>
          )}
        </p>

        <Math display>{'y = \\sigma(X \\cdot W + b)'}</Math>

        <div className="space-y-2 text-xs text-muted-foreground my-4 font-mono">
          <p><span className="text-foreground font-bold">X (Batch × InFeatures):</span> {t('Girdi veri matrisi', 'Input batch feature matrix')}</p>
          <p><span className="text-primary font-bold">W (InFeatures × OutFeatures):</span> {t('Katmanın öğrenilebilir ağırlık matrisi', 'Trainable weight matrix')}</p>
          <p><span className="text-accent font-bold">b (1 × OutFeatures):</span> {t('Yayınlanan (broadcast) bias vektörü', 'Broadcast bias vector')}</p>
          <p><span className="text-emerald-400 font-bold">σ (Aktivasyon):</span> {t('Doğrusal olmayan fonksiyon (ReLU, GeLU, Sigmoid)', 'Non-linear activation (ReLU, GeLU, Sigmoid)')}</p>
        </div>

        <SubHeading>{t('Geri Yayılım (Backpropagation) ve Matris Gradyanları', 'Backpropagation & Matrix Gradients')}</SubHeading>
        <p>
          {lang === 'tr' ? (
            <>
              Model eğitilirken kayıp fonksiyonunun (<Math>{'L'}</Math>) ağırlıklara göre kısmi türevi Zincir Kuralı ile hesaplanır.
              Bir matris çarpımının gradyanı yine bir matris çarpımıdır:
            </>
          ) : (
            <>
              During training, the gradient of the scalar loss <Math>{'L'}</Math> with respect to the weight matrix is derived
              via the multivariable chain rule. The gradient of a matrix product is itself a matrix product:
            </>
          )}
        </p>

        <Math display>{'\\frac{\\partial L}{\\partial W} = X^T \\cdot \\frac{\\partial L}{\\partial Y}'}</Math>

        <p className="text-xs text-muted-foreground">
          {t(
            'MatrixFlash-Pro Autograd motoru (Variable / VarTensor), bu matris çarpımı gradyanlarını GPU üzerinde arka planda otomatik olarak hesaplar!',
            'MatrixFlash-Pro’s Autograd tape engine automatically evaluates these transposed matrix gradients on CUDA during backward()!',
          )}
        </p>
      </DocSection>

      {/* BÖLÜM 6 */}
      <DocSection id="bolum-6" title={t('Bölüm 6: GPU Mimarisi & CUDA Neden Matris Sever?', 'Chapter 6: GPU Architecture & Why CUDA Loves Matrices')}>
        <p>
          {lang === 'tr' ? (
            <>
              Bir CPU, karmaşık mantıksal kararlar ve sıralı kodlar için tasarlanmış birkaç güçlü çekirdeğe (ör. 8-16 çekirdek) sahiptir.
              Buna karşılık modern bir NVIDIA GPU, aynı anda çalışan <strong className="text-primary">on binlerce hafif çekirdeğe (CUDA Cores &amp; Tensor Cores)</strong> sahiptir.
            </>
          ) : (
            <>
              A modern CPU is designed for complex branching and low latency on sequential tasks with a few heavy cores (8–16).
              In stark contrast, an NVIDIA GPU packs <strong className="text-primary">tens of thousands of concurrent parallel threads</strong>{' '}
              optimized for high arithmetic throughput.
            </>
          )}
        </p>

        <div className="my-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-border/80 bg-card/60 p-5">
            <Cpu className="size-6 text-primary mb-3" />
            <h4 className="font-mono text-sm font-bold text-foreground">{t('SIMT Modeli', 'SIMT Execution')}</h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'Single Instruction Multiple Threads: 32 iş parçacığı (Warp) aynı talimatı farklı matris hücreleri üzerinde paralel olarak yürütür.',
                'Single Instruction Multiple Threads: 32 threads in a Warp execute the same instruction concurrently on different matrix elements.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/60 p-5">
            <Layers className="size-6 text-accent mb-3" />
            <h4 className="font-mono text-sm font-bold text-foreground">{t('Shared Memory Tiling', 'Shared Memory Tiling')}</h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'Yavaş global bellek yerine, matris blokları çip üzerindeki hızlı Shared Memory tamponuna kopyalanır ve tekrar tekrar kullanılır.',
                'Sub-matrices are cached into ultra-fast on-chip Shared Memory, slashing slow DRAM accesses by orders of magnitude.',
              )}
            </p>
          </div>

          <div className="rounded-xl border border-border/80 bg-card/60 p-5">
            <Zap className="size-6 text-emerald-400 mb-3" />
            <h4 className="font-mono text-sm font-bold text-foreground">{t('TF32 Tensor Cores', 'TF32 Tensor Cores')}</h4>
            <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
              {t(
                'Donanımsal 4×4 matris çarpma-toplama (MMA) üniteleri tek saat çevriminde devasa matris hesaplamalarını tamamlar.',
                'Hardware 4x4 matrix-multiply-accumulate (MMA) units execute dense linear algebra in a single clock cycle.',
              )}
            </p>
          </div>
        </div>

        <Callout type="tip" title={t('Özet Ders Notu', 'Course Summary Takeaway')}>
          {t(
            'Matris = Koordinat uzayının lineer dönüşüm kuralı. GPU = Bu dönüşümü binlerce hücrede aynı anda hesaplayan paralel fabrika. MatrixFlash-Pro = Bu muazzam gücü C++17 zarafetiyle parmaklarınızın ucuna getiren köprüdür.',
            'Matrix = The geometric transformation rule of space. GPU = The massively parallel engine evaluating that rule across millions of numbers simultaneously. MatrixFlash-Pro = The bridge uniting that raw power with modern C++17 elegance.',
          )}
        </Callout>

        <div className="mt-8 flex flex-wrap items-center justify-between border-t border-border/80 pt-6">
          <Link
            href="/docs"
            className="text-xs font-mono text-muted-foreground hover:text-foreground"
          >
            ← {t('Giriş & Kurulum', 'Getting Started')}
          </Link>
          <Link
            href="/docs/api"
            className="inline-flex items-center gap-2 text-sm font-semibold text-primary hover:underline"
          >
            {t('Sonraki: Eksiksiz API Referansı', 'Next: Complete API Reference')}
            <ArrowRight className="size-4" />
          </Link>
        </div>
      </DocSection>
    </article>
  )
}

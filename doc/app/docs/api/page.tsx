import { ApiEntry } from '../../../components/docs/api-entry'
import { DocSection, InlineCode, PageHeader, Callout } from '../../../components/docs/doc-ui'

export default function ApiPage() {
  return (
    <article>
      <PageHeader
        eyebrow="Referans"
        title="API Referansı"
        description="MatrixFlash-Pro'nun sunduğu tüm oluşturucular, operasyonlar ve yardımcı fonksiyonlar. Her giriş imza, açıklama ve örnek içerir."
      />

      <Callout type="info" title="Ad alanı">
        Tüm tipler <InlineCode>matrix_pro</InlineCode> ad alanı altındadır.
        Örneklerde <InlineCode>using matrix_pro::Matrix;</InlineCode> yazıldığı
        varsayılmıştır.
      </Callout>

      {/* OLUŞTURUCULAR */}
      <DocSection id="olusturucular" title="Oluşturucular (Factories)">
        <p>
          Bir matrisi baştan üretmenin çeşitli yolları. Statik fabrika
          metotları, verilen boyutta yeni bir matrisi doğrudan GPU belleğinde
          oluşturur.
        </p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="Matrix{...}"
          signature="Matrix(std::initializer_list<std::initializer_list<float>> data)"
          badge="constructor"
          params={[
            {
              name: 'data',
              type: 'initializer_list',
              desc: 'Satır satır matris değerleri. İç içe süslü parantezlerle verilir.',
            },
          ]}
          returns="Matrix"
          example={`Matrix a{{1.0f, 2.0f},
         {3.0f, 4.0f}};   // 2x2, otomatik GPU'ya yüklenir`}
        >
          Host verisinden doğrudan bir matris oluşturur ve değerleri anında
          device belleğine kopyalar.
        </ApiEntry>

        <ApiEntry
          name="Matrix::zeros"
          signature="static Matrix zeros(size_t rows, size_t cols)"
          badge="static"
          params={[
            { name: 'rows', type: 'size_t', desc: 'Satır sayısı.' },
            { name: 'cols', type: 'size_t', desc: 'Sütun sayısı.' },
          ]}
          returns="Matrix"
          example={`Matrix z = Matrix::zeros(3, 3);  // tüm elemanlar 0`}
        >
          Tüm elemanları <InlineCode>0</InlineCode> olan bir matris üretir.
        </ApiEntry>

        <ApiEntry
          name="Matrix::ones"
          signature="static Matrix ones(size_t rows, size_t cols)"
          badge="static"
          params={[
            { name: 'rows', type: 'size_t', desc: 'Satır sayısı.' },
            { name: 'cols', type: 'size_t', desc: 'Sütun sayısı.' },
          ]}
          returns="Matrix"
          example={`Matrix o = Matrix::ones(2, 4);   // tüm elemanlar 1`}
        >
          Tüm elemanları <InlineCode>1</InlineCode> olan bir matris üretir.
        </ApiEntry>

        <ApiEntry
          name="Matrix::identity"
          signature="static Matrix identity(size_t n)"
          badge="static"
          params={[{ name: 'n', type: 'size_t', desc: 'Kare matrisin boyutu.' }]}
          returns="Matrix"
          example={`Matrix I = Matrix::identity(3);
// 1 0 0
// 0 1 0
// 0 0 1`}
        >
          Köşegeni <InlineCode>1</InlineCode>, diğer elemanları{' '}
          <InlineCode>0</InlineCode> olan <InlineCode>n×n</InlineCode> birim
          matris üretir. Matris çarpımının etkisiz elemanıdır.
        </ApiEntry>

        <ApiEntry
          name="Matrix::flatten"
          signature="Matrix flatten() const"
          badge="static"
          params={[
            { name: 'rows', type: 'size_t', desc: 'Satır sayısı.' },
            { name: 'cols', type: 'size_t', desc: 'Sütun sayısı.' },
          ]}
          returns="Matrix"
          example={`Matrix w = Matrix::ones(4, 4);
Matrix flat = w.flatten();  // 16x1`}
        >
          Matrisi tek sütunlu bir vektöre dönüştürür. Veri akışlarında ve
          aktivasyon öncesi hazırlıkta kullanışlıdır.
        </ApiEntry>
      </div>

      {/* ELEMENTWISE */}
      <DocSection id="elementwise" title="Elementwise İşlemler">
        <p>
          Karşılıklı elemanlar üzerinde çalışan operasyonlar. Operatörler ve
          metotlar yeni bir <InlineCode>Matrix</InlineCode> döndürür.
        </p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="operator+ / operator-"
          signature="Matrix operator+(const Matrix& other) const"
          params={[
            {
              name: 'other',
              type: 'const Matrix&',
              desc: 'Aynı boyutta ikinci matris.',
            },
          ]}
          returns="Matrix"
          example={`Matrix c = a + b;   // elementwise toplama
Matrix d = a - b;   // elementwise çıkarma`}
        >
          İki eşit boyutlu matrisi karşılıklı elemanlarını toplayarak/çıkararak
          birleştirir.
        </ApiEntry>

        <ApiEntry
          name="hadamard"
          signature="Matrix hadamard(const Matrix& other) const"
          params={[
            {
              name: 'other',
              type: 'const Matrix&',
              desc: 'Aynı boyutta ikinci matris.',
            },
          ]}
          returns="Matrix"
          example={`Matrix c = a.hadamard(b);  // eleman-eleman çarpım`}
        >
          Hadamard (elementwise) çarpım. Matris çarpımından farklı olarak,
          karşılıklı elemanlar tek tek çarpılır.
        </ApiEntry>

        <ApiEntry
          name="scale"
          signature="Matrix scale(float c) const"
          params={[
            { name: 'c', type: 'float', desc: 'Çarpılacak skaler değer.' },
          ]}
          returns="Matrix"
          example={`Matrix half = a.scale(0.5f);  // tüm elemanlar * 0.5`}
        >
          Matrisin her elemanını verilen skaler ile çarpar.
        </ApiEntry>
      </div>

      {/* BROADCAST */}
      <DocSection id="broadcast" title="Yayınlama (Broadcasting)">
        <p>
          Farklı boyuttaki bir vektörün, bir matrisin her satırına veya sütununa
          otomatik olarak uygulanmasıdır. Sinir ağlarında bias (yanlılık) ekleme
          işlemi için sık kullanılır.
        </p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="broadcast_add"
          signature="Matrix broadcast_add(const Matrix& vec) const"
          params={[
            {
              name: 'vec',
              type: 'const Matrix&',
              desc: 'Her satıra eklenecek satır vektörü (1×n).',
            },
          ]}
          returns="Matrix"
          example={`Matrix x = Matrix::randn(4, 3);
Matrix bias{{0.1f, 0.2f, 0.3f}};  // 1x3
Matrix y = x.broadcast_add(bias); // her satıra eklenir`}
        >
          Bir satır vektörünü matrisin tüm satırlarına ekler. Vektör, matrisin
          satır sayısı kadar tekrarlanmış gibi davranır.
        </ApiEntry>
      </div>

      {/* MATMUL zaten matris nedir'de ama referansta da */}
      <DocSection title="Matris Çarpımı">
        <p>
          En maliyetli işlem olan matris çarpımı, arka planda{' '}
          shared-memory tiled CUDA kernel tarafından yürütülür.
        </p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="operator*"
          signature="Matrix operator*(const Matrix& other) const"
          badge="CUDA"
          params={[
            {
              name: 'other',
              type: 'const Matrix&',
              desc: 'Boyutları uyumlu ikinci matris (m×k · k×n).',
            },
          ]}
          returns="Matrix"
          example={`Matrix a = Matrix::ones(128, 256);
Matrix b = Matrix::ones(256, 64);
Matrix c = a * b;   // 128x64, tiled CUDA ile`}
        >
          Standart matris çarpımı. Sol matrisin sütun sayısı, sağ matrisin satır
          sayısına eşit olmalıdır.
        </ApiEntry>

        <ApiEntry
          name="outer_product"
          signature="Matrix outer_product(const Matrix& other) const"
          params={[
            {
              name: 'other',
              type: 'const Matrix&',
              desc: 'İkinci vektör.',
            },
          ]}
          returns="Matrix"
          example={`Matrix u{{1.0f}, {2.0f}, {3.0f}};  // 3x1
Matrix v{{4.0f, 5.0f}};            // 1x2
Matrix m = u.outer_product(v);     // 3x2`}
        >
          İki vektörün dış çarpımını hesaplayarak bir matris üretir.
        </ApiEntry>
      </div>

      {/* İNDİRGEMELER */}
      <DocSection id="indirgemeler" title="İndirgemeler (Reductions)">
        <p>
          Bir matrisin elemanlarını tek bir değere veya bir eksen boyunca daha
          küçük bir matrise indirgeyen istatistiksel operasyonlar. Tümü GPU
          üzerinde paralel indirgeme (parallel reduction) ile hesaplanır.
        </p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="sum"
          signature="float sum() const"
          returns="float"
          example={`float total = a.sum();  // tüm elemanların toplamı`}
        >
          Matristeki tüm elemanların toplamını döndürür.
        </ApiEntry>
        <ApiEntry
          name="mean"
          signature="float mean() const"
          returns="float"
          example={`float avg = a.mean();  // ortalama değer`}
        >
          Tüm elemanların aritmetik ortalamasını döndürür.
        </ApiEntry>
        <ApiEntry
          name="max / min"
          signature="float max() const  |  float min() const"
          returns="float"
          example={`float hi = a.max();
float lo = a.min();`}
        >
          Matristeki en büyük ve en küçük elemanı döndürür.
        </ApiEntry>
      </div>

      {/* DÖNÜŞÜM & AKTİVASYON */}
      <DocSection id="donusum" title="Dönüşüm & Aktivasyon">
        <p>
          Şekil dönüşümleri ve sinir ağlarında kullanılan aktivasyon
          fonksiyonları. Aktivasyonlar doğrudan GPU üzerinde uygulanır.
        </p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="transpose"
          signature="Matrix transpose() const"
          returns="Matrix"
          example={`Matrix at = a.transpose();  // satır <-> sütun`}
        >
          Matrisin devriğini alır. <InlineCode>m×n</InlineCode> matris,{' '}
          <InlineCode>n×m</InlineCode> matrise dönüşür.
        </ApiEntry>
        <ApiEntry
          name="relu"
          signature="Matrix relu() const"
          badge="activation"
          returns="Matrix"
          example={`Matrix y = x.relu();  // max(0, x)`}
        >
          Rectified Linear Unit. Negatif değerleri sıfırlar, pozitifleri korur:{' '}
          <InlineCode>max(0, x)</InlineCode>.
        </ApiEntry>
        <ApiEntry
          name="sigmoid"
          signature="Matrix sigmoid() const"
          badge="activation"
          returns="Matrix"
          example={`Matrix y = x.sigmoid();  // 1 / (1 + e^-x)`}
        >
          Değerleri <InlineCode>(0, 1)</InlineCode> aralığına sıkıştıran sigmoid
          fonksiyonu.
        </ApiEntry>
        <ApiEntry
          name="tanh"
          signature="Matrix tanh() const"
          badge="activation"
          returns="Matrix"
          example={`Matrix y = x.tanh();  // (-1, 1) aralığı`}
        >
          Hiperbolik tanjant. Değerleri <InlineCode>(-1, 1)</InlineCode> aralığına
          sıkıştırır.
        </ApiEntry>
        <ApiEntry
          name="softmax"
          signature="Matrix softmax() const"
          badge="activation"
          returns="Matrix"
          example={`Matrix probs = logits.softmax();
// çıktı toplamı 1.0 olan olasılık dağılımı`}
        >
          Bir vektörü olasılık dağılımına dönüştürür. Taşmaya (overflow) karşı{' '}
          <strong className="text-foreground">sayısal kararlı</strong> şekilde
          implemente edilmiştir.
        </ApiEntry>
      </div>

      {/* İLERİ */}
      <DocSection id="ileri" title="İleri Doğrusal Cebir">
        <p>Kare matrisler üzerinde çalışan gelişmiş operasyonlar.</p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="determinant"
          signature="float determinant() const"
          returns="float"
          example={`Matrix a{{4.0f, 7.0f},
         {2.0f, 6.0f}};
float det = a.determinant();  // 10.0`}
        >
          Kare bir matrisin determinantını hesaplar.
        </ApiEntry>
        <ApiEntry
          name="inverse"
          signature="Matrix inverse() const"
          returns="Matrix"
          example={`Matrix inv = a.inverse();  // a * inv = I`}
        >
          Kare bir matrisin tersini hesaplar. Tersi ile çarpımı birim matrisi
          verir. Determinantı sıfır olan matrislerin tersi yoktur.
        </ApiEntry>
        <ApiEntry
          name="solve"
          signature="Matrix solve(const Matrix& rhs) const"
          returns="Matrix"
          example={`Matrix x = a.solve(b);  // a * x = b`}
        >
          Ax = b doğrusal sistemini LU ayrıştırması (getrf + getrs) ile,
          <code>inverse()</code> almadan çözer. Daha hızlı ve sayısal olarak
          kararlıdır. <code>rhs</code> vektör veya birden çok sağ taraf olabilir.
        </ApiEntry>
        <ApiEntry
          name="qr"
          signature="QRResult qr() const"
          returns="QRResult{q,r}"
          example={`auto [q, r] = a.qr();  // a = q * r`}
        >
          İnce (thin) QR ayrıştırması (geqrf + orgqr). Ortogonalleştirme ve
          en küçük kareler problemleri için kullanılır.
        </ApiEntry>
        <ApiEntry
          name="svd"
          signature="SVDResult svd() const"
          returns="SVDResult{u,s,v}"
          example={`auto res = a.svd();  // a = u * s * v^T`}
        >
          Tekil değer ayrıştırması (gesvd). PCA, boyut indirgeme ve düşük
          ranklı yaklaşıklamanın temelidir.
        </ApiEntry>
        <ApiEntry
          name="cholesky"
          signature="Matrix cholesky() const"
          returns="Matrix"
          example={`Matrix l = a.cholesky();  // a = l * l^T`}
        >
          Pozitif tanımlı simetrik matrisler için Cholesky ayrıştırması (potrf).
          Alt üçgensel L faktörünü döndürür.
        </ApiEntry>
        <ApiEntry
          name="eigen"
          signature="EigenResult eigen() const"
          returns="EigenResult{eigenvalues,eigenvectors}"
          example={`auto res = a.eigen();`}
        >
          Simetrik matrisler için özdeğer/özvektör ayrıştırması (syevd).
          Özdeğerler köşegen, özvektörler sütunlar halinde döner.
        </ApiEntry>
        <ApiEntry
          name="pinv"
          signature="Matrix pinv() const"
          returns="Matrix"
          example={`Matrix p = a.pinv();  // Moore-Penrose`}
        >
          Kare olmayan matrislerin SVD tabanlı Moore-Penrose pseudo-tersini
          hesaplar.
        </ApiEntry>
        <ApiEntry
          name="rank"
          signature="std::size_t rank() const"
          returns="std::size_t"
          example={`std::size_t r = a.rank();`}
        >
          SVD tekil değerleri üzerinden matrisin sayısal rankını döndürür.
        </ApiEntry>
        <ApiEntry
          name="solve_least_squares"
          signature="Matrix solve_least_squares(const Matrix& rhs) const"
          returns="Matrix"
          example={`Matrix x = a.solve_least_squares(b);`}
        >
          Aşırı/aşağı belirlenmiş sistemlerin SVD pseudo-tersi üzerinden en
          küçük kareler çözümü (x = pinv(A) * b).
        </ApiEntry>
      </div>

      {/* KALICILIK & YAŞAM DÖNGÜSÜ */}
      <DocSection id="kalicilik" title="Kalıcılık & Yaşam Döngüsü">
        <p>
          Matrisi diske kaydetme/yükleme ve host↔device veri transferini yöneten
          fonksiyonlar.
        </p>
      </DocSection>
      <div className="space-y-5">
        <ApiEntry
          name="download"
          signature="void download()"
          returns="void"
          example={`Matrix c = a * b;   // GPU'da
c.download();       // sonucu host belleğine indir`}
        >
          Device belleğindeki sonucu host (CPU) belleğine kopyalar. Sonuca
          erişmeden veya yazdırmadan önce çağrılmalıdır.
        </ApiEntry>
        <ApiEntry
          name="print"
          signature="void print() const"
          returns="void"
          example={`c.download();
c.print();  // matrisi konsola yazdırır`}
        >
          Matrisi okunabilir bir biçimde konsola yazdırır.
        </ApiEntry>
        <ApiEntry
          name="save / load"
          signature="void save(const std::string& path)  |  static Matrix load(const std::string& path)"
          badge="I/O"
          params={[
            {
              name: 'path',
              type: 'const std::string&',
              desc: 'Dosya yolu.',
            },
          ]}
          returns="void / Matrix"
          example={`a.save("weights.bin");
Matrix loaded = Matrix::load("weights.bin");`}
        >
          Bir matrisi ikili (binary) biçimde diske kaydeder ve tekrar yükler.
          Eğitilmiş ağırlıkları saklamak için kullanışlıdır.
        </ApiEntry>
      </div>
    </article>
  )
}

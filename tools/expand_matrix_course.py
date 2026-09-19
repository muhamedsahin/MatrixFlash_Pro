#!/usr/bin/env python3
"""Prepend absolute-beginner chapters to matrix-course.json and refresh meta."""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "doc" / "content" / "matrix-course.json"


def L(tr, en):
    return {"tr": tr, "en": en}


def beginner_sections():
    return [
        {
            "id": "bolum-0",
            "title": L("0. Matris Nedir? (Sıfırdan)", "0. What Is a Matrix? (From Zero)"),
            "intro": L(
                "Matris, sayılardan oluşan bir **dikdörtgen tablodur**. Excel sayfası, "
                "telefon fotoğrafındaki pikseller, bir sınıftaki not listesi — hepsi matrisi "
                "andırır. Bu bölüm hiç matematik geçmişi olmadan başlar.",
                "A matrix is a **rectangular table of numbers**. A spreadsheet, the pixels "
                "in a phone photo, a class grade list — all look like matrices. This chapter "
                "starts with zero math background.",
            ),
            "blocks": [
                {
                    "type": "p",
                    "text": L(
                        "Günlük hayatta sürekli tablolar kullanırız: satırlar bir kişiyi, "
                        "sütunlar bir özelliği (yaş, not, fiyat) tutar. Matematikte bu tabloya "
                        "**matris** denir ve her kutuya bir **eleman** denir.",
                        "We use tables every day: rows hold one person, columns hold one "
                        "feature (age, grade, price). In math that table is a **matrix**, and "
                        "each cell is an **entry**.",
                    ),
                },
                {
                    "type": "figure",
                    "data": [["2", "5", "1"], ["0", "3", "8"]],
                    "caption": L(
                        "2 satır × 3 sütunluk bir matris. Okunuş: 'ikiye üç matris'.",
                        "A 2-row × 3-column matrix. Spoken: 'a two-by-three matrix'.",
                    ),
                    "accent": "accent",
                },
                {
                    "type": "list",
                    "items": [
                        L(
                            "**Satır (row):** soldan sağa giden yatay çizgi.",
                            "**Row:** the horizontal line left → right.",
                        ),
                        L(
                            "**Sütun (column):** yukarıdan aşağı giden dikey çizgi.",
                            "**Column:** the vertical line top → bottom.",
                        ),
                        L(
                            "**Şekil (shape):** `(satır sayısı, sütun sayısı)` — örn. `(2, 3)`.",
                            "**Shape:** `(rows, cols)` — e.g. `(2, 3)`.",
                        ),
                        L(
                            "**İndeks:** `A[i, j]` = i. satır, j. sütundaki sayı (genelde 0'dan başlar).",
                            "**Index:** `A[i, j]` = number in row i, column j (usually 0-based).",
                        ),
                    ],
                },
                {
                    "type": "callout",
                    "variant": "tip",
                    "title": L("Gerçek dünya örnekleri", "Real-world examples"),
                    "text": L(
                        "Gri bir fotoğraf: her piksel bir sayı → matris. RGB fotoğraf: üç matris "
                        "(kırmızı, yeşil, mavi). Ses spektrogramı, tavsiye sistemindeki puan "
                        "tablosu, sinir ağındaki ağırlıklar — hepsi matris.",
                        "A grayscale photo: each pixel is a number → a matrix. An RGB photo: "
                        "three matrices. Audio spectrograms, recommender score tables, neural "
                        "network weights — all matrices.",
                    ),
                },
                {
                    "type": "callout",
                    "variant": "info",
                    "title": L("Kütüphanede", "In the library"),
                    "text": L(
                        "`Matrix a(2, 3);` 2×3 boş matris açar. `a.at(0, 1) = 5;` ilk satırın "
                        "ikinci elemanını yazar. GPU'da hesap için `upload()` / sonuç için `download()`.",
                        "`Matrix a(2, 3);` creates a 2×3 matrix. `a.at(0, 1) = 5;` writes the "
                        "second entry of the first row. Use `upload()` before GPU work and "
                        "`download()` to read results on the host.",
                    ),
                },
            ],
        },
        {
            "id": "bolum-0b",
            "title": L(
                "0b. Toplama, Skaler Çarpma, Basit Kurallar",
                "0b. Addition, Scalar Multiply, Simple Rules",
            ),
            "intro": L(
                "İki matrisi toplamak için **aynı şekilde** olmaları gerekir. Bir sayıyla "
                "çarpma (skaler) ise her kutuyu o sayıyla çarpmaktır — tıpkı her fiyata "
                "%10 zam yapmak gibi.",
                "To add two matrices they must have the **same shape**. Multiplying by a "
                "number (a scalar) multiplies every entry — like raising every price by 10%.",
            ),
            "blocks": [
                {
                    "type": "math",
                    "display": True,
                    "tex": "(A + B)_{ij} = A_{ij} + B_{ij}, \\qquad (cA)_{ij} = c \\, A_{ij}",
                },
                {
                    "type": "p",
                    "text": L(
                        "Örnek: `[[1, 2], [3, 4]] + [[10, 0], [0, 1]] = [[11, 2], [3, 5]]`. "
                        "Şekil uymuyorsa işlem tanımsızdır — kütüphane `ShapeMismatchError` fırlatır.",
                        "Example: `[[1, 2], [3, 4]] + [[10, 0], [0, 1]] = [[11, 2], [3, 5]]`. "
                        "If shapes disagree the op is undefined — the library throws "
                        "`ShapeMismatchError`.",
                    ),
                },
                {
                    "type": "list",
                    "items": [
                        L(
                            "**Elementwise çarpım (`*` değil, `elementwise_multiply`):** aynı "
                            "konumdaki kutular çarpılır (Hadamard).",
                            "**Elementwise product (not matmul):** multiply matching cells "
                            "(Hadamard product).",
                        ),
                        L(
                            "**Matris çarpımı (`A * B`):** tamamen farklı kural — bir sonraki "
                            "başlangıç bölümü ve Bölüm 2.",
                            "**Matrix product (`A * B`):** a different rule — next beginner "
                            "chapter and Chapter 2.",
                        ),
                    ],
                },
                {
                    "type": "code",
                    "lang": "cpp",
                    "filename": "beginner_add.cpp",
                    "code": (
                        '#include "matrix_pro/matrix_pro.hpp"\n'
                        "using namespace matrix_pro;\n\n"
                        "Matrix a{{1, 2}, {3, 4}};\n"
                        "Matrix b{{10, 0}, {0, 1}};\n"
                        "Matrix c = a + b;          // [[11, 2], [3, 5]]\n"
                        "Matrix d = a * 2.0f;       // her eleman × 2\n"
                        "c.download(); d.download();"
                    ),
                },
            ],
        },
        {
            "id": "bolum-0c",
            "title": L(
                "0c. Matris × Vektör: En Sezgisel Çarpım",
                "0c. Matrix × Vector: The Most Intuitive Product",
            ),
            "intro": L(
                "Bir matrisi bir vektörle çarpmak, vektörün her bileşenini matrisin "
                "**sütunlarıyla karıştırmaktır**. Düşün: 3 ürünün fiyatı (vektör) × "
                "her müşterinin aldığı adet tablosu (matris) → müşteri başına toplam tutar.",
                "Multiplying a matrix by a vector mixes the vector's components with the "
                "matrix **columns**. Think: prices of 3 products (vector) × how many each "
                "customer bought (matrix) → total per customer.",
            ),
            "blocks": [
                {
                    "type": "math",
                    "display": True,
                    "tex": "A x = x_1 a_{:1} + x_2 a_{:2} + \\cdots + x_n a_{:n}",
                },
                {
                    "type": "p",
                    "text": L(
                        "`A` boyutu `m×n`, `x` boyutu `n×1` ise sonuç `m×1` vektördür. "
                        "İç boyutlar (`n`) eşleşmek zorundadır — tıpkı fişteki ürün sayısıyla "
                        "fiyat listesinin eşleşmesi gibi.",
                        "If `A` is `m×n` and `x` is `n×1`, the result is `m×1`. The inner "
                        "sizes (`n`) must match — like matching product counts to a price list.",
                    ),
                },
                {
                    "type": "figure",
                    "data": [["2", "0"], ["1", "3"]],
                    "caption": L(
                        "2×2 matris. Vektör `[1, 4]` ile çarpım: `1·[2,1] + 4·[0,3] = [2, 13]`.",
                        "2×2 matrix. Times vector `[1, 4]`: `1·[2,1] + 4·[0,3] = [2, 13]`.",
                    ),
                    "accent": "primary",
                },
                {
                    "type": "callout",
                    "variant": "tip",
                    "title": L("Neden GPU?", "Why GPU?"),
                    "text": L(
                        "Binlerce satır × binlerce sütun olduğunda bu karışımlar trilyonlarca "
                        "çarpma ister. MatrixFlash-Pro bunu NVIDIA GPU + cuBLAS ile saniyenin "
                        "kesirinde yapar (bkz. Performans sayfası).",
                        "With thousands of rows and columns these mixes need trillions of "
                        "multiplies. MatrixFlash-Pro does that in a fraction of a second on "
                        "an NVIDIA GPU via cuBLAS (see the Performance page).",
                    ),
                },
            ],
        },
        {
            "id": "bolum-0d",
            "title": L(
                "0d. Matris × Matris: Kapıyı Açan İşlem",
                "0d. Matrix × Matrix: The Door-Opening Op",
            ),
            "intro": L(
                "İki matrisi çarpmak, 'önce bir dönüşüm, sonra diğeri' demektir. "
                "Boyut kuralı basit: `(m×K) · (K×n) → (m×n)`. Ortadaki `K` eşleşmezse "
                "çarpım yapılamaz.",
                "Multiplying two matrices means 'apply one transform, then the other'. "
                "The size rule is simple: `(m×K)·(K×n)→(m×n)`. If the middle `K` disagrees, "
                "the product is illegal.",
            ),
            "blocks": [
                {
                    "type": "math",
                    "display": True,
                    "tex": "(AB)_{ij} = \\sum_{k=1}^{K} A_{ik} B_{kj}",
                },
                {
                    "type": "p",
                    "text": L(
                        "Her sonuç kutusu bir **nokta çarpımdır**: A'nın bir satırı ile B'nin "
                        "bir sütunu. Bu yüzden toplam iş `2·m·K·n` flopluktur — kütüphanenin "
                        "GFLOPS ölçümünün kaynağı.",
                        "Every output cell is a **dot product**: one row of A with one column "
                        "of B. Total work is `2·m·K·n` flops — the source of the library's "
                        "GFLOPS numbers.",
                    ),
                },
                {
                    "type": "code",
                    "lang": "cpp",
                    "filename": "beginner_matmul.cpp",
                    "code": (
                        "Matrix A{{1, 2, 3}, {4, 5, 6}};      // 2×3\n"
                        "Matrix B{{7, 8}, {9, 10}, {11, 12}}; // 3×2\n"
                        "Matrix C = A * B;   // 2×2, GPU GEMM\n"
                        "// veya tahsissiz hot path:\n"
                        "Matrix Out(2, 2, MemoryMode::device_only);\n"
                        "multiply_into(A, B, Out);"
                    ),
                },
                {
                    "type": "callout",
                    "variant": "info",
                    "title": L("Sıradaki yol", "Where next"),
                    "text": L(
                        "Sezgi tamamsa Bölüm 1'den itibaren üniversite düzeyine geçersin: "
                        "vektör uzayları, rank, özdeğer, SVD. Hiçbir şey kaçmaz — sadece "
                        "dil biraz daha resmi olur.",
                        "Once the intuition clicks, Chapter 1 onward is university level: "
                        "vector spaces, rank, eigenvalues, SVD. Nothing is skipped — the "
                        "language just gets more formal.",
                    ),
                },
            ],
        },
    ]


def main():
    data = json.loads(PATH.read_text(encoding="utf-8"))
    existing_ids = {s["id"] for s in data["sections"]}
    beginners = [s for s in beginner_sections() if s["id"] not in existing_ids]
    data["sections"] = beginners + data["sections"]
    data["meta"] = {
        "eyebrow": L("SIFIRDAN ÜNİVERSİTEYE MATRİS DERSİ", "MATRIX COURSE FROM ZERO TO UNIVERSITY"),
        "title": L(
            "Matris Nedir? Lineer Cebirden Derin Öğrenmeye",
            "What Is a Matrix? From Linear Algebra to Deep Learning",
        ),
        "description": L(
            "Önce günlük dilde matris: tablo, piksel, fiyat listesi. Sonra vektör "
            "uzayları, çarpım, determinant, rank, özdeğer, SVD ve en küçük kareler. "
            "Her bölüm MatrixFlash-Pro API'sine bağlanır.",
            "First matrices in plain language: tables, pixels, price lists. Then vector "
            "spaces, products, determinants, rank, eigenvalues, SVD and least squares. "
            "Every chapter links to the MatrixFlash-Pro API.",
        ),
    }
    PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sections:", [s["id"] for s in data["sections"]])


if __name__ == "__main__":
    main()

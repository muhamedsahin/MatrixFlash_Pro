import json
from pathlib import Path

OUT = (Path(__file__).resolve().parents[1] / "doc" / "content" / "docs" /
       "benchmarks-comparison.json")


def L(tr, en):
    return {"tr": tr, "en": en}


page = {
    "slug": "benchmarks-comparison",
    "meta": {
        "eyebrow": L("PERFORMANS KARSILASTIRMA", "PERFORMANCE COMPARISON"),
        "title": L("MatrixFlash-Pro vs Benzer Araclar",
                   "MatrixFlash-Pro vs Similar Tools"),
        "description": L(
            "Ayni makinede olculmus GEMM ve MLP baz cizgileri: "
            "MatrixFlash-Pro, ham cuBLAS, naive CUDA cekirdegi, "
            "tek-thread CPU ve autograd bant maliyeti.",
            "Same-machine GEMM and MLP baselines: MatrixFlash-Pro, raw cuBLAS, "
            "a naive CUDA kernel, single-thread CPU and the autograd tape cost."),
    },
    "sections": [],
}

sec_method = {
    "id": "metodoloji",
    "title": L("Metodoloji ve Adillik Kurallari", "Methodology and Fairness Rules"),
    "intro": L(
        "Tum 'olculdu' satirlari RTX 3070 Laptop GPU (CC 8.6, 8 GiB) uzerinde, "
        "release preset ile alindi. GPU: CUDA-event medyan, CPU: chrono "
        "wall-clock medyan. Her vaka 3 warmup + 10 tekrar.",
        "Every 'measured' row comes from an RTX 3070 Laptop GPU (CC 8.6, 8 GiB), "
        "release preset. GPU: CUDA-event median, CPU: chrono wall-clock median. "
        "Each case: 3 warmup + 10 repeats."),
    "blocks": [
        {"type": "list", "items": [
            L("Olculdu: ayni makinede bu repo ile tekrar uretilebilir.", "Measured: reproducible on the same machine with this repo."),
            L("Referans (harici): PyTorch / ArrayFire / OpenBLAS degerleri literaturdendir.", "Reference (external): PyTorch / ArrayFire / OpenBLAS figures are from the literature."),
            L("mflash operator*: ciktiyi tahsis eder (gercek maliyet).", "mflash operator*: allocates its output (real cost)."),
            L("raw cuBLAS: onceden tahsisli tamponda kosar (saf cekirdek hizi).", "raw cuBLAS: runs on pre-allocated buffers (pure kernel speed)."),
            L("CPU naive yalnizca n <= 512 icin kosar.", "CPU naive runs only for n <= 512."),
        ]},
        {"type": "callout", "variant": "warn",
         "title": L("Sayilari dogru okuyun", "Read the numbers correctly"),
         "text": L("GFLOPS yuksek-iyi, ms dusuk-iyi metriktir. Ham cuBLAS satirindaki asiri yuksek GFLOPS, tahsissiz mikro-benchmark etkisidir.",
                   "GFLOPS is higher-is-better, ms is lower-is-better. The very high GFLOPS on the raw cuBLAS row is the allocation-free micro-benchmark effect.")},
        {"type": "code", "lang": "bash", "filename": "terminal",
         "code": "cmake --preset release\ncmake --build --preset release --target matrix_pro_bench_comparison\n./build/benchmarks/Release/matrix_pro_bench_comparison.exe --sizes 256,512,1024,2048 --repeats 10 --warmup 3 --csv benchmarks/results/comparison.csv --json benchmarks/results/comparison.json\npython tools/sync_benchmark_data.py"},
    ],
}
page["sections"].append(sec_method)
def T(headers_tr_en):
    return [{"tr": a, "en": b} for (a, b) in headers_tr_en]


sec_gemm = {
    "id": "gemm",
    "title": L("GEMM: Kare Matris Carpimi (olculdu)", "GEMM: Square Matmul (measured)"),
    "intro": L(
        "RTX 3070 Laptop, 19.09.2026 olcumu. mflash = Matrix::operator*, "
        "raw-cuBLAS = onceden tahsisli cublasGemmEx, naive-gpu = ilk-deneme "
        "cekirdegi, cpu-naive = tek-thread uclu dongu.",
        "RTX 3070 Laptop, measured 2026-09-19. mflash = Matrix::operator*, "
        "raw-cuBLAS = pre-allocated cublasGemmEx, naive-gpu = first-attempt "
        "kernel, cpu-naive = single-thread triple loop."),
    "blocks": [
        {"type": "table",
         "headers": T([("Vaka (NxN)", "Case (NxN)"),
                       ("mflash (ms / GFLOPS)", "mflash (ms / GFLOPS)"),
                       ("ham cuBLAS (ms / GFLOPS)", "raw cuBLAS (ms / GFLOPS)"),
                       ("naive GPU (ms / GFLOPS)", "naive GPU (ms / GFLOPS)"),
                       ("CPU naive (ms / GFLOPS)", "CPU naive (ms / GFLOPS)")]),
         "rows": [
            [L("256x256 olculdu", "256x256 measured"),
             L("0.041 ms / 819 GFLOPS", "0.041 ms / 819 GFLOPS"),
             L("0.032 ms / 1040 GFLOPS", "0.032 ms / 1040 GFLOPS"),
             L("0.050 ms / 676 GFLOPS", "0.050 ms / 676 GFLOPS"),
             L("15.02 ms / 2.23 GFLOPS", "15.02 ms / 2.23 GFLOPS")],
            [L("512x512 olculdu", "512x512 measured"),
             L("0.345 ms / 779 GFLOPS", "0.345 ms / 779 GFLOPS"),
             L("0.052 ms / 5140 GFLOPS", "0.052 ms / 5140 GFLOPS"),
             L("0.300 ms / 896 GFLOPS", "0.300 ms / 896 GFLOPS"),
             L("105.83 ms / 2.54 GFLOPS", "105.83 ms / 2.54 GFLOPS")],
            [L("1024x1024 olculdu", "1024x1024 measured"),
             L("0.751 ms / 2859 GFLOPS", "0.751 ms / 2859 GFLOPS"),
             L("0.195 ms / 11038 GFLOPS", "0.195 ms / 11038 GFLOPS"),
             L("2.230 ms / 963 GFLOPS", "2.230 ms / 963 GFLOPS"),
             L("atlandi (O(n3) host)", "skipped (O(n3) host)")],
            [L("2048x2048 olculdu", "2048x2048 measured"),
             L("3.049 ms / 5635 GFLOPS", "3.049 ms / 5635 GFLOPS"),
             L("1.238 ms / 13877 GFLOPS", "1.238 ms / 13877 GFLOPS"),
             L("atlandi (O(n3) thread)", "skipped (O(n3) threads)"),
             L("atlandi (O(n3) host)", "skipped (O(n3) host)")],
         ]},
        {"type": "callout", "variant": "tip",
         "title": L("Tablo ne soyluyor", "What the table says"),
         "text": L("CPU'ya karsi ~300-1100x hizlanma. Naive GPU cekirdegine karsi 1024'te ~3x. mflash ile ham cuBLAS farki cogunlukla cikti tahsisi + wrapper maliyetidir.",
                   "300-1100x over CPU. About 3x over the naive GPU kernel at 1024. The mflash vs raw cuBLAS gap is mostly output allocation + wrapper cost.")},
    ],
}
page["sections"].append(sec_gemm)
sec_mlp = {
    "id": "mlp-tape",
    "title": L("Egitim Adimi: Autograd Bant Maliyeti (olculdu)",
               "Training Step: Autograd Tape Cost (measured)"),
    "intro": L("2 katmanli MLP (256 giris, 10 sinif, batch=64). mlp-full = ileri + geri, mlp-fwd = yalnizca ileri.",
               "2-layer MLP (256 inputs, 10 classes, batch=64). mlp-full = forward + backward, mlp-fwd = forward only."),
    "blocks": [
        {"type": "table",
         "headers": T([("Gizli genislik", "Hidden width"),
                       ("mlp-full (ms / samples/s)", "mlp-full (ms / samples/s)"),
                       ("mlp-fwd (ms / samples/s)", "mlp-fwd (ms / samples/s)"),
                       ("Bant maliyeti", "Tape cost")]),
         "rows": [
            [L("h=256", "h=256"), L("1.745 ms / 36678", "1.745 ms / 36678"),
             L("1.278 ms / 50080", "1.278 ms / 50080"), L("1.4x", "1.4x")],
            [L("h=512", "h=512"), L("2.078 ms / 30796", "2.078 ms / 30796"),
             L("1.364 ms / 46922", "1.364 ms / 46922"), L("1.5x", "1.5x")],
            [L("h=1024", "h=1024"), L("2.951 ms / 21686", "2.951 ms / 21686"),
             L("1.323 ms / 48375", "1.323 ms / 48375"), L("2.2x", "2.2x")],
            [L("h=2048", "h=2048"), L("4.152 ms / 15413", "4.152 ms / 15413"),
             L("1.693 ms / 37799", "1.693 ms / 37799"), L("2.5x", "2.5x")],
         ]},
        {"type": "p",
         "text": L("Bant maliyeti 1.4-2.5x araligindadir. Cikarim yolunda Variable degil Matrix kullanin.",
                   "The tape cost is in the 1.4-2.5x range. Use Matrix, not Variable, on the inference path.")},
    ],
}
page["sections"].append(sec_mlp)
sec_ext = {
    "id": "harici-referans",
    "title": L("Harici Referanslar (olculmedi)", "External References (not measured)"),
    "intro": L("Asagidaki satirlar bu makinede olculmemistir; kaba baglam icin literatur degerleridir.",
               "The rows below were NOT measured on this machine; literature figures for rough context."),
    "blocks": [
        {"type": "table",
         "headers": T([("Arac", "Tool"), ("Islem", "Operation"),
                       ("Kaba deger", "Rough figure"), ("Kaynak", "Source")]),
         "rows": [
            [L("PyTorch (CUDA) referans", "PyTorch (CUDA) reference"),
             L("torch.mm fp32, benzer Ampere GPU", "torch.mm fp32, similar Ampere GPU"),
             L("cuBLAS tabanli; ham-cuBLAS bandinda", "cuBLAS-based; in raw-cuBLAS band"),
             L("pytorch.org/docs", "pytorch.org/docs")],
            [L("ArrayFire (CUDA) referans", "ArrayFire (CUDA) reference"),
             L("matmul fp32, benzer Ampere GPU", "matmul fp32, similar Ampere GPU"),
             L("cuBLAS tabanli", "cuBLAS-based"),
             L("arrayfire.org/docs", "arrayfire.org/docs")],
            [L("Eigen (CPU) referans", "Eigen (CPU) reference"),
             L("cok-thread MatrixXf carpimi", "multi-thread MatrixXf product"),
             L("10-30 GFLOPS bandi", "10-30 GFLOPS band"),
             L("eigen.tuxfamily.org", "eigen.tuxfamily.org")],
            [L("OpenBLAS (CPU) referans", "OpenBLAS (CPU) reference"),
             L("sgemm cok-thread", "sgemm multi-thread"),
             L("10-40 GFLOPS bandi", "10-40 GFLOPS band"),
             L("openblas.net", "openblas.net")],
         ]},
        {"type": "callout", "variant": "info",
         "title": L("Eigen neden olculmedi", "Why Eigen is not measured"),
         "text": L("Eigen basligi derleme aninda bulunursa bench otomatik olcer. Bu makinede kurulu degildi.",
                   "The bench measures Eigen automatically when the header is visible at build time. It was not installed here.")},
    ],
}
page["sections"].append(sec_ext)
sec_how = {
    "id": "nasil-tekrarlanir",
    "title": L("Nasil Tekrarlanir", "How to Reproduce"),
    "intro": L("Tek komutla ayni CSV/JSON'u uretin, sonra doc verisini senkronlayin.",
               "Produce the same CSV/JSON with one command, then sync the doc data."),
    "blocks": [
        {"type": "steps", "items": [
            {"title": L("Karsilastirma benchmarkini derleyin", "Build the comparison benchmark"),
             "text": L("cmake --preset release + target matrix_pro_bench_comparison.",
                       "cmake --preset release + target matrix_pro_bench_comparison.")},
            {"title": L("Olcumu alin", "Run the measurement"),
             "text": L("--sizes 256,512,1024,2048 --repeats 10 --warmup 3 ile calistirin.",
                       "Run with --sizes 256,512,1024,2048 --repeats 10 --warmup 3.")},
            {"title": L("Doc verisini senkronlayin", "Sync the doc data"),
             "text": L("python tools/sync_benchmark_data.py calistirin.",
                       "Run python tools/sync_benchmark_data.py.")},
        ]},
        {"type": "code", "lang": "bash", "filename": "terminal",
         "code": "cmake --preset release\ncmake --build --preset release --target matrix_pro_bench_comparison\n./build/benchmarks/Release/matrix_pro_bench_comparison.exe --sizes 256,512,1024,2048 --repeats 10 --warmup 3 --csv benchmarks/results/comparison.csv --json benchmarks/results/comparison.json\npython tools/sync_benchmark_data.py"},
    ],
}
page["sections"].append(sec_how)
# __SECTIONS__





OUT.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", OUT, len(page["sections"]), "sections")



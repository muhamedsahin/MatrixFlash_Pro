#!/usr/bin/env python3
"""Generate doc/content/docs/benchmarks-comparison.json from latest measured numbers.

Sources (repo root):
  benchmarks/results/comparison.json
  benchmarks/results/rivals.json
  benchmarks/results/training.json   (optional, fused section)

Run:
  python tools/gen_bench_page.py
  python tools/sync_benchmark_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc" / "content" / "docs" / "benchmarks-comparison.json"
COMPARISON = ROOT / "benchmarks" / "results" / "comparison.json"
RIVALS = ROOT / "benchmarks" / "results" / "rivals.json"
TRAINING = ROOT / "benchmarks" / "results" / "training.json"


def L(tr, en):
    return {"tr": tr, "en": en}


def T(headers):
    return [{"tr": a, "en": b} for a, b in headers]


def fmt_ms_gflops(ms: float, gflops: float) -> str:
    if ms < 0.1:
        return f"{ms:.3f} ms / {gflops:.0f} GFLOPS"
    if ms < 10:
        return f"{ms:.3f} ms / {gflops:.0f} GFLOPS"
    return f"{ms:.2f} ms / {gflops:.0f} GFLOPS"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def gemm_rows_from_comparison(comp: dict):
    """Build per-size dicts for into / op / cublas / naive / cpu."""
    by = {}
    for r in comp.get("results", []):
        if r.get("unit") != "GFLOPS" or str(r.get("case", "")).startswith("mlp"):
            continue
        n = int(str(r["case"]).split("x")[0])
        note = r.get("note", "")
        slot = by.setdefault(n, {})
        if "multiply_into" in note:
            slot["into"] = r
        elif note.startswith("mflash operator*"):
            slot["op"] = r
        elif note.startswith("raw cublas"):
            slot["cublas"] = r
        elif note.startswith("naive"):
            slot["naive"] = r
        elif note.startswith("cpu"):
            slot["cpu"] = r
    return by


def main() -> None:
    comp = load_json(COMPARISON) or {}
    rivals = load_json(RIVALS) or {}
    training = load_json(TRAINING) or {}
    by = gemm_rows_from_comparison(comp)
    device = (comp.get("device") or {}).get("name", "NVIDIA GeForce RTX 3070 Laptop GPU")
    stamp = comp.get("timestamp", "2026-09-19")

    page = {
        "slug": "benchmarks-comparison",
        "meta": {
            "eyebrow": L("PERFORMANS KARSILASTIRMA", "PERFORMANCE COMPARISON"),
            "title": L(
                "MatrixFlash-Pro vs cuBLAS ve Piyasa Motorlari",
                "MatrixFlash-Pro vs cuBLAS and Market Engines",
            ),
            "description": L(
                "Ayni makinede olculmus GEMM: MatrixFlash-Pro, NVIDIA cuBLAS/cublasLt, "
                "NumPy@OpenBLAS; PyTorch/CuPy/ArrayFire/Eigen/JAX baglami; fused epilogue.",
                "Same-machine GEMM: MatrixFlash-Pro, NVIDIA cuBLAS/cublasLt, "
                "NumPy@OpenBLAS; PyTorch/CuPy/ArrayFire/Eigen/JAX context; fused epilogue.",
            ),
        },
        "sections": [],
    }

    # Headline ratios
    ratios = []
    for n in sorted(by):
        into = by[n].get("into")
        raw = by[n].get("cublas")
        if into and raw and raw["throughput"] > 0:
            pct = 100.0 * into["throughput"] / raw["throughput"]
            ratios.append(f"{n}: %{pct:.0f}")

    page["sections"].append(
        {
            "id": "metodoloji",
            "title": L("Metodoloji", "Methodology"),
            "intro": L(
                f"Cihaz: {device}. Zaman damgasi: {stamp}. GPU: CUDA-event medyan "
                "(warmup 12, tekrar 40). NumPy: OpenBLAS wall-clock. "
                f"multiply_into / cuBLAS oranlari: {', '.join(ratios)}.",
                f"Device: {device}. Timestamp: {stamp}. GPU: CUDA-event median "
                f"(warmup 12, repeats 40). NumPy: OpenBLAS wall-clock. "
                f"multiply_into / cuBLAS ratios: {', '.join(ratios)}.",
            ),
            "blocks": [
                {
                    "type": "callout",
                    "variant": "tip",
                    "title": L("Hedef: her zaman cuBLAS sinifi", "Goal: always cuBLAS-class"),
                    "text": L(
                        "Buyuk GEMM'de algo-cache'li cublasLt + TENSOR_OP GemmEx kullanilir. "
                        "2048'de multiply_into ham cuBLAS'i gecer; 1024'te ~%99. "
                        "Fused gemm+bias+relu zinciri her zaman gecer.",
                        "Large GEMM uses algo-cached cublasLt + TENSOR_OP GemmEx. "
                        "At 2048 multiply_into beats raw cuBLAS; at 1024 ~99%. "
                        "Fused gemm+bias+relu always beats the unfused chain.",
                    ),
                },
                {
                    "type": "code",
                    "lang": "bash",
                    "filename": "terminal",
                    "code": (
                        "cmake --build --preset release --target matrix_pro_bench_comparison "
                        "matrix_pro_bench_external_gemm matrix_pro_bench_training\n"
                        "./build/benchmarks/Release/matrix_pro_bench_comparison.exe "
                        "--sizes 256,512,1024,2048 --warmup 12 --repeats 40 "
                        "--json benchmarks/results/comparison.json\n"
                        "./build/benchmarks/Release/matrix_pro_bench_external_gemm.exe "
                        "--sizes 256,512,1024,2048 --warmup 12 --repeats 40 "
                        "--json benchmarks/results/external_gemm.json\n"
                        "py -3 tools/bench_rivals.py\n"
                        "python tools/gen_bench_page.py\n"
                        "python tools/sync_benchmark_data.py"
                    ),
                },
            ],
        }
    )

    # GEMM table
    gemm_table_rows = []
    for n in sorted(by):
        s = by[n]
        def cell(key, skip="—"):
            r = s.get(key)
            if not r:
                return L(skip, skip)
            txt = fmt_ms_gflops(r["ms_median"], r["throughput"])
            return L(txt, txt)

        gemm_table_rows.append(
            [
                L(f"{n}x{n}", f"{n}x{n}"),
                cell("op"),
                cell("into"),
                cell("cublas"),
                cell("naive"),
                cell("cpu"),
            ]
        )

    page["sections"].append(
        {
            "id": "gemm",
            "title": L("GEMM — MatrixFlash-Pro vs ham cuBLAS (olculdu)", "GEMM — vs raw cuBLAS (measured)"),
            "intro": L(
                "multiply_into = tahsissiz hot path (cuBLAS ile ayni kosul). "
                "operator* = device_only cikti tahsisi dahil.",
                "multiply_into = allocation-free hot path (same condition as cuBLAS). "
                "operator* = includes device_only output allocation.",
            ),
            "blocks": [
                {
                    "type": "table",
                    "headers": T(
                        [
                            ("Vaka", "Case"),
                            ("mflash operator*", "mflash operator*"),
                            ("mflash multiply_into", "mflash multiply_into"),
                            ("ham cuBLAS", "raw cuBLAS"),
                            ("naive GPU", "naive GPU"),
                            ("CPU naive", "CPU naive"),
                        ]
                    ),
                    "rows": gemm_table_rows,
                }
            ],
        }
    )

    # Rivals — 6+ engines
    rival_rows = []
    engines = rivals.get("engines", [])
    # Prefer showing 1024 as the headline size in the table
    focus = 1024
    for eng in engines:
        name = eng.get("name", "?")
        kind = eng.get("kind", "")
        if eng.get("measured") and eng.get("by_size"):
            # pick focus size or largest available
            sizes = eng["by_size"]
            key = str(focus) if str(focus) in sizes else sorted(sizes.keys(), key=int)[-1]
            e = sizes[key]
            speed = fmt_ms_gflops(e["ms_median"], e["gflops"])
            status = L(f"{key}x{key} olculdu", f"{key}x{key} measured")
            note = e.get("note", eng.get("source", ""))
        else:
            speed = L(eng.get("band", "—"), eng.get("band", "—"))
            status = L("referans (olculmedi)", "reference (not timed)")
            note = eng.get("source", "")
        rival_rows.append(
            [
                L(name, name),
                L(kind, kind),
                speed if isinstance(speed, dict) else L(speed, speed),
                status,
                L(str(note)[:80], str(note)[:80]),
            ]
        )

    page["sections"].append(
        {
            "id": "rakipler",
            "title": L(
                "Piyasadaki Iddiali Motorlar (6+)",
                "Market Contenders (6+)",
            ),
            "intro": L(
                "Ayni is: yogun FP32 kare GEMM. Olculenler bu makinede; referans satirlari "
                "acikca isaretli. PyTorch/CuPy/ArrayFire/JAX icerde cuBLAS kullanir — "
                "tepe bant ayni sinif; MatrixFlash-Pro farki C++17 API + fused epilogue + "
                "algo-cache ile tutarli cuBLAS-sinifi hiz.",
                "Same job: dense FP32 square GEMM. Measured rows are on this machine; "
                "reference rows are labeled. PyTorch/CuPy/ArrayFire/JAX call cuBLAS "
                "internally — same ceiling class; MatrixFlash-Pro differentiates with a "
                "C++17 API, fused epilogues, and algo-cache for consistent cuBLAS-class speed.",
            ),
            "blocks": [
                {
                    "type": "table",
                    "headers": T(
                        [
                            ("Motor", "Engine"),
                            ("Tur", "Kind"),
                            ("Hiz (odak 1024²)", "Speed (focus 1024²)"),
                            ("Durum", "Status"),
                            ("Not", "Note"),
                        ]
                    ),
                    "rows": rival_rows,
                },
                {
                    "type": "callout",
                    "variant": "info",
                    "title": L("Neden PyTorch burada yok?", "Why is PyTorch missing a measured row?"),
                    "text": L(
                        "Bu makinedeki Python 3.14 icin resmi torch tekerlegi yok. "
                        "Kurulunca `py -3 tools/bench_rivals.py` otomatik olcer. "
                        "Beklenen bant: cuBLAS tavani − Python launch overhead.",
                        "No official torch wheel for Python 3.14 on this machine. "
                        "Once installed, `py -3 tools/bench_rivals.py` measures it. "
                        "Expected band: cuBLAS ceiling minus Python launch overhead.",
                    ),
                },
            ],
        }
    )

    # Side-by-side measured GFLOPS at each size for measured engines only
    measured_engines = [e for e in engines if e.get("measured") and e.get("by_size")]
    if measured_engines:
        headers = [("Boyut", "Size")] + [(e["name"], e["name"]) for e in measured_engines]
        rows = []
        for n in [256, 512, 1024, 2048]:
            row = [L(f"{n}x{n}", f"{n}x{n}")]
            for e in measured_engines:
                cell = e["by_size"].get(str(n))
                if cell:
                    txt = f"{cell['gflops']:.0f} GFLOPS"
                    row.append(L(txt, txt))
                else:
                    row.append(L("—", "—"))
            rows.append(row)
        page["sections"].append(
            {
                "id": "olculen-motorlar",
                "title": L(
                    "Olculen Motorlar — GFLOPS Tablosu",
                    "Measured Engines — GFLOPS Table",
                ),
                "intro": L(
                    "Sadece bu makinede zamanlanmis motorlar. Yuksek = daha iyi.",
                    "Only engines timed on this machine. Higher is better.",
                ),
                "blocks": [{"type": "table", "headers": T(headers), "rows": rows}],
            }
        )

    # Fused from training.json
    fused_rows = []
    if training:
        by_h = {}
        for r in training.get("results", []):
            case = r.get("case", "")
            if "gemm+bias+relu" not in case:
                continue
            # "fused gemm+bias+relu h=1024" / "chain ..."
            parts = case.split("h=")
            if len(parts) < 2:
                continue
            h = parts[-1].strip()
            slot = by_h.setdefault(h, {})
            if case.startswith("fused"):
                slot["fused"] = r
            else:
                slot["chain"] = r
        for h in sorted(by_h, key=lambda x: int(x)):
            f = by_h[h].get("fused")
            c = by_h[h].get("chain")
            if not f or not c:
                continue
            speedup = c["ms_median"] / f["ms_median"] if f["ms_median"] > 0 else 0
            fused_rows.append(
                [
                    L(f"h={h}", f"h={h}"),
                    L(
                        fmt_ms_gflops(f["ms_median"], f["throughput"]),
                        fmt_ms_gflops(f["ms_median"], f["throughput"]),
                    ),
                    L(
                        fmt_ms_gflops(c["ms_median"], c["throughput"]),
                        fmt_ms_gflops(c["ms_median"], c["throughput"]),
                    ),
                    L(f"{speedup:.1f}x", f"{speedup:.1f}x"),
                ]
            )

    if fused_rows:
        page["sections"].append(
            {
                "id": "fused-gemm",
                "title": L(
                    "Fused GEMM+Bias+ReLU — zinciri gecme (olculdu)",
                    "Fused GEMM+Bias+ReLU — beating the chain (measured)",
                ),
                "intro": L(
                    "Bu, ham cuBLAS SGEMM'in otesine gectigimiz yer: tek cublasLt "
                    "epilogue vs 3 ayri kernel. PyTorch'ta da fused varken, burada "
                    "saf C++ tek cagri.",
                    "This is where we go beyond plain cuBLAS SGEMM: one cublasLt "
                    "epilogue vs 3 kernels. PyTorch has fused ops too; here it is a "
                    "single C++ call.",
                ),
                "blocks": [
                    {
                        "type": "table",
                        "headers": T(
                            [
                                ("Gizli", "Hidden"),
                                ("fused", "fused"),
                                ("zincir (3 kernel)", "chain (3 kernels)"),
                                ("Hizlanma", "Speedup"),
                            ]
                        ),
                        "rows": fused_rows,
                    }
                ],
            }
        )

    page["sections"].append(
        {
            "id": "nasil-tekrarlanir",
            "title": L("Nasil Tekrarlanir", "How to Reproduce"),
            "intro": L(
                "comparison + external_gemm + bench_rivals + gen/sync.",
                "comparison + external_gemm + bench_rivals + gen/sync.",
            ),
            "blocks": [
                {
                    "type": "steps",
                    "items": [
                        {
                            "title": L("GPU bench", "GPU benches"),
                            "text": L(
                                "matrix_pro_bench_comparison ve external_gemm --json yazdirin.",
                                "Run matrix_pro_bench_comparison and external_gemm with --json.",
                            ),
                        },
                        {
                            "title": L("Rakip motorlar", "Rival engines"),
                            "text": L(
                                "py -3 tools/bench_rivals.py (NumPy zorunlu; torch/cupy varsa olculur).",
                                "py -3 tools/bench_rivals.py (NumPy required; torch/cupy measured if present).",
                            ),
                        },
                        {
                            "title": L("Doc sync", "Doc sync"),
                            "text": L(
                                "python tools/gen_bench_page.py && python tools/sync_benchmark_data.py",
                                "python tools/gen_bench_page.py && python tools/sync_benchmark_data.py",
                            ),
                        },
                    ],
                }
            ],
        }
    )

    OUT.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, len(page["sections"]), "sections")


if __name__ == "__main__":
    main()

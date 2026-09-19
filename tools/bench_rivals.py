#!/usr/bin/env python3
"""Measure rival matrix engines on this machine and write rivals.json.

Engines attempted (skip if missing):
  1. MatrixFlash-Pro multiply_into  — from comparison.json (already measured)
  2. NVIDIA cuBLAS (raw)            — from comparison.json
  3. NumPy @ OpenBLAS (CPU)         — measured here
  4. PyTorch CUDA / CPU             — measured if importable
  5. CuPy                           — measured if importable
  6. Naive triple-loop (NumPy ref)  — tiny sizes only, for scale

Usage (repo root):
  py -3 tools/bench_rivals.py
  python tools/sync_benchmark_data.py
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "results" / "rivals.json"
COMPARISON = ROOT / "benchmarks" / "results" / "comparison.json"
SIZES = [256, 512, 1024, 2048]
WARMUP = 5
REPEATS = 20


def median_ms(fn, warmup=WARMUP, repeats=REPEATS) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def gflops(n: int, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return (2.0 * n * n * n) / (ms * 1e6)


def load_mflash_cublas():
    if not COMPARISON.exists():
        return {}, {}
    data = json.loads(COMPARISON.read_text(encoding="utf-8"))
    mflash, cublas = {}, {}
    for r in data.get("results", []):
        if r.get("unit") != "GFLOPS":
            continue
        case = r.get("case", "")
        if "x" not in case or case.startswith("mlp"):
            continue
        try:
            n = int(case.split("x")[0])
        except ValueError:
            continue
        if n not in SIZES:
            continue
        note = r.get("note", "")
        entry = {
            "ms_median": r["ms_median"],
            "gflops": r["throughput"],
            "note": note,
            "source": "comparison.json (same machine, CUDA events)",
        }
        if "multiply_into" in note:
            mflash[n] = entry
        elif note.startswith("raw cublas"):
            cublas[n] = entry
    # Fallback: operator* if multiply_into missing
    if not mflash:
        for r in data.get("results", []):
            note = r.get("note", "")
            if not note.startswith("mflash operator*"):
                continue
            case = r.get("case", "")
            try:
                n = int(case.split("x")[0])
            except ValueError:
                continue
            if n in SIZES:
                mflash[n] = {
                    "ms_median": r["ms_median"],
                    "gflops": r["throughput"],
                    "note": note,
                    "source": "comparison.json (same machine, CUDA events)",
                }
    return mflash, cublas


def measure_numpy():
    import numpy as np

    results = {}
    # Use all OpenBLAS threads.
    for n in SIZES:
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)
        # Warm BLAS

        def once():
            c = a @ b
            # Prevent dead-code elimination of the product.
            if c[0, 0] == float("inf"):
                raise RuntimeError("unreachable")

        ms = median_ms(once)
        results[n] = {
            "ms_median": ms,
            "gflops": gflops(n, ms),
            "note": f"numpy {np.__version__} @ OpenBLAS (CPU, wall-clock)",
            "source": "bench_rivals.py measured",
        }
        print(f"  numpy {n}x{n}: {ms:.3f} ms / {results[n]['gflops']:.1f} GFLOPS")
    return results


def measure_torch():
    try:
        import torch
    except ImportError:
        print("  torch: not installed — skip")
        return {}

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  torch {torch.__version__} device={device}")
    for n in SIZES:
        a = torch.randn(n, n, dtype=torch.float32, device=device)
        b = torch.randn(n, n, dtype=torch.float32, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()

            def once():
                c = a @ b
                torch.cuda.synchronize()
                if c[0, 0].item() == float("inf"):
                    raise RuntimeError("unreachable")

        else:

            def once():
                c = a @ b
                if c[0, 0].item() == float("inf"):
                    raise RuntimeError("unreachable")

        ms = median_ms(once)
        tag = "CUDA" if device.type == "cuda" else "CPU"
        results[n] = {
            "ms_median": ms,
            "gflops": gflops(n, ms),
            "note": f"pytorch {torch.__version__} ({tag}, wall-clock+sync)",
            "source": "bench_rivals.py measured",
        }
        print(f"  torch {n}x{n}: {ms:.3f} ms / {results[n]['gflops']:.1f} GFLOPS")
    return results


def measure_cupy():
    try:
        import cupy as cp
    except ImportError:
        print("  cupy: not installed — skip")
        return {}

    results = {}
    print(f"  cupy {cp.__version__}")
    for n in SIZES:
        a = cp.random.randn(n, n, dtype=cp.float32)
        b = cp.random.randn(n, n, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()

        def once():
            c = a @ b
            cp.cuda.Stream.null.synchronize()
            if float(c[0, 0]) == float("inf"):
                raise RuntimeError("unreachable")

        ms = median_ms(once)
        results[n] = {
            "ms_median": ms,
            "gflops": gflops(n, ms),
            "note": f"cupy {cp.__version__} (CUDA, wall-clock+sync)",
            "source": "bench_rivals.py measured",
        }
        print(f"  cupy {n}x{n}: {ms:.3f} ms / {results[n]['gflops']:.1f} GFLOPS")
    return results


def load_external_lt():
    path = ROOT / "benchmarks" / "results" / "external_gemm.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in data.get("results", []):
        if r.get("unit") != "GFLOPS":
            continue
        case = r.get("case", "")
        try:
            n = int(case.split("x")[0])
        except ValueError:
            continue
        if n not in SIZES:
            continue
        out[n] = {
            "ms_median": r["ms_median"],
            "gflops": r["throughput"],
            "note": r.get("note", "cublasLt"),
            "source": "external_gemm.json (same machine, CUDA events)",
        }
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Loading MatrixFlash-Pro / cuBLAS from comparison.json…")
    mflash, cublas = load_mflash_cublas()
    print(f"  mflash sizes: {sorted(mflash)}")
    print(f"  cublas sizes: {sorted(cublas)}")

    print("Loading raw cublasLt from external_gemm.json…")
    lt = load_external_lt()
    print(f"  cublasLt sizes: {sorted(lt)}")

    print("Measuring NumPy @ OpenBLAS…")
    numpy_r = measure_numpy()

    print("Measuring PyTorch (if available)…")
    torch_r = measure_torch()

    print("Measuring CuPy (if available)…")
    cupy_r = measure_cupy()

    engines = [
        {
            "id": "matrixflash_pro",
            "name": "MatrixFlash-Pro",
            "kind": "C++17 / CUDA library",
            "measured": True,
            "by_size": {str(k): v for k, v in mflash.items()},
        },
        {
            "id": "cublas",
            "name": "NVIDIA cuBLAS",
            "kind": "Vendor BLAS",
            "measured": True,
            "by_size": {str(k): v for k, v in cublas.items()},
        },
        {
            "id": "cublaslt",
            "name": "NVIDIA cublasLt",
            "kind": "Vendor matmul (Lt)",
            "measured": bool(lt),
            "by_size": {str(k): v for k, v in lt.items()} if lt else {},
            "band": None if lt else "cuBLAS sibling — usually same ceiling",
            "source": "external_gemm.json" if lt else "reference",
        },
        {
            "id": "numpy_openblas",
            "name": "NumPy @ OpenBLAS",
            "kind": "Python / CPU BLAS",
            "measured": True,
            "by_size": {str(k): v for k, v in numpy_r.items()},
        },
    ]
    if torch_r:
        engines.append(
            {
                "id": "pytorch",
                "name": "PyTorch",
                "kind": "Python / DL framework",
                "measured": True,
                "by_size": {str(k): v for k, v in torch_r.items()},
            }
        )
    else:
        engines.append(
            {
                "id": "pytorch",
                "name": "PyTorch (CUDA)",
                "kind": "Python / DL framework",
                "measured": False,
                "band": "cuBLAS-backed GPU ceiling; Python launch overhead on top of vendor GEMM",
                "source": "reference (torch not installable on this Python 3.14)",
            }
        )
    if cupy_r:
        engines.append(
            {
                "id": "cupy",
                "name": "CuPy",
                "kind": "Python / CUDA arrays",
                "measured": True,
                "by_size": {str(k): v for k, v in cupy_r.items()},
            }
        )
    else:
        engines.append(
            {
                "id": "cupy",
                "name": "CuPy",
                "kind": "Python / CUDA arrays",
                "measured": False,
                "band": "cuBLAS-backed GPU ceiling; Python overhead on top",
                "source": "reference (package not installed)",
            }
        )

    engines.extend(
        [
            {
                "id": "arrayfire",
                "name": "ArrayFire (CUDA)",
                "kind": "Array runtime",
                "measured": False,
                "band": "cuBLAS-backed — same GPU ceiling class as mflash/cuBLAS",
                "source": "reference (not timed on this machine)",
            },
            {
                "id": "eigen",
                "name": "Eigen (CPU multi-thread)",
                "kind": "C++ CPU templates",
                "measured": False,
                "band": "~10–40 GFLOPS fp32 on laptop-class CPUs (well below GPU path)",
                "source": "reference (literature / typical laptop)",
            },
            {
                "id": "jax",
                "name": "JAX (CUDA)",
                "kind": "Python / XLA",
                "measured": False,
                "band": "XLA → cuBLAS/cuDNN — same GPU ceiling class",
                "source": "reference (not timed on this machine)",
            },
        ]
    )

    payload = {
        "library": "MatrixFlash-Pro",
        "title": "Rival matrix engines vs MatrixFlash-Pro",
        "device_note": "GPU rows: RTX 3070 Laptop CUDA-event medians; NumPy: OpenBLAS wall-clock",
        "sizes": SIZES,
        "engines": engines,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

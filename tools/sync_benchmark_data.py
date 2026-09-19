# Syncs measured benchmark JSON into the docs site.
# Usage (PowerShell, repo root):
#   python tools/sync_benchmark_data.py
# Copies benchmarks/results/*.json -> doc/public/data/benchmarks/*.json
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmarks" / "results"
DST = ROOT / "doc" / "public" / "data" / "benchmarks"

COPY_FILES = [
    "comparison.json",
    "matmul.json",
    "training.json",
    "rivals.json",
    "external_gemm.json",
]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        src = SRC / name
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        # Validate JSON before copying so a half-written run never ships.
        json.loads(src.read_text(encoding="utf-8"))
        shutil.copy2(src, DST / name)
        print(f"synced: {name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Soft-label JSONL from a teacher GGUF via `ereshkigal` / `semif-score`.

Vulkan: N_GPU_LAYERS=99 (Pascal 1080 Ti). Never reuse training/.venv for SemIf A/B.

Usage:
  ERESHKIGAL_GGUF=models/Qwen3.5-4B-Q4_K_M.gguf \\
    python training/teacher_label.py --in fixtures/lang/dev.jsonl --out training/teacher_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gguf", type=Path, default=os.environ.get("ERESHKIGAL_GGUF"))
    p.add_argument("--bin", default=os.environ.get("ERESHKIGAL_BIN", "semif-score"))
    args = p.parse_args()
    if args.gguf is None:
        print("need --gguf or ERESHKIGAL_GGUF", file=sys.stderr)
        return 2
    bin_path = shutil.which(args.bin) or args.bin
    cmd = [bin_path, "--gguf", str(args.gguf), "--in", str(args.inp), "--out", str(args.out)]
    env = os.environ.copy()
    print("teacher", " ".join(cmd), "N_GPU_LAYERS=" + env.get("N_GPU_LAYERS", "0"))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        return r.returncode
    n = sum(1 for _ in args.out.open() if _.strip())
    print(f"wrote {args.out} n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

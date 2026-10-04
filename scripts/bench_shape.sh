#!/usr/bin/env bash
# Shared vs direct on fixtures/shape15.jsonl (CPU appendix). GPU: set N_GPU_LAYERS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${BIN:-$ROOT/target/release/semif-score}"
GGUF="${1:-${ERESHKIGAL_GGUF:-models/Qwen3-0.6B-Q8_0.gguf}}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
REV="${REV:-c1899de289a04d12100db370d81485cdf75e47ca}"
THREADS="${LLAMA_THREADS:-8}"
GPU="${N_GPU_LAYERS:-0}"
INPUT="${INPUT:-fixtures/shape15.jsonl}"
mkdir -p results
"$BIN" --mode direct --gguf "$GGUF" --model "$MODEL" --revision "$REV" \
  --input "$INPUT" --output results/shape_direct.jsonl \
  --llama-threads "$THREADS" --n-gpu-layers "$GPU" --no-replay --no-radix
"$BIN" --mode shared --gguf "$GGUF" --model "$MODEL" --revision "$REV" \
  --input "$INPUT" --output results/shape_shared.jsonl \
  --llama-threads "$THREADS" --n-gpu-layers "$GPU" --n-seq-max 16
python3 - <<PY
import json, time
from pathlib import Path
d=[json.loads(l) for l in Path("results/shape_direct.jsonl").read_text().splitlines() if l.strip()]
s=[json.loads(l) for l in Path("results/shape_shared.jsonl").read_text().splitlines() if l.strip()]
dt=sum(r.get("total_seconds") or 0 for r in d)
# shared rows may omit total; use file mtime not reliable — print lens
rep={"n": len(d), "direct_sum_total_s": dt, "shared_rows": len(s),
     "gpu_layers": int("$GPU"),
     "note": "CPU appendix unless N_GPU_LAYERS>0. shape777: download SemIf fixture separately."}
Path("results/shape_bench.json").write_text(json.dumps(rep, indent=2)+"\n")
print(json.dumps(rep, indent=2))
PY

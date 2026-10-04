#!/usr/bin/env bash
# Fair same-GGUF A/B: Ereshkigal vs optional Python SemIf llamacpp.
# Usage: ./scripts/ab_llamacpp.sh [gguf] [input.jsonl]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GGUF="${1:-models/Qwen_Qwen3.5-4B-Q4_K_M.gguf}"
INPUT="${2:-examples/decisions.jsonl}"
BIN="${BIN:-$ROOT/target/release/semif-score}"
THREADS="${LLAMA_THREADS:-8}"
GPU="${N_GPU_LAYERS:-0}"
mkdir -p results
OUT_E="results/ab_ereshkigal.jsonl"
"$BIN" --mode direct --gguf "$GGUF" --input "$INPUT" --output "$OUT_E" \
  --llama-threads "$THREADS" --n-gpu-layers "$GPU" --no-replay \
  --model "${MODEL:-Qwen/Qwen3.5-4B}" --revision "${REV:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
if [[ -n "${SEMIF_SCORE:-}" ]]; then
  OUT_P="results/ab_semif_llamacpp.jsonl"
  "$SEMIF_SCORE" --mode direct --backend llamacpp --gguf "$GGUF" \
    --input "$INPUT" --output "$OUT_P" --llama-threads "$THREADS"
  python3 - <<PY
import json
from pathlib import Path
a=[json.loads(l) for l in Path("$OUT_E").read_text().splitlines() if l.strip()]
b=[json.loads(l) for l in Path("$OUT_P").read_text().splitlines() if l.strip()]
by={r["id"]:r for r in b}
agree=0; dps=[]
for r in a:
    o=by[r["id"]]
    ai=max(range(len(r["probabilities"])), key=lambda i: r["probabilities"][i])
    bi=max(range(len(o["probabilities"])), key=lambda i: o["probabilities"][i])
    agree += ai==bi
    dps.append(max(abs(x-y) for x,y in zip(r["probabilities"], o["probabilities"])))
n=len(a)
rep={
  "n": n,
  "argmax_agree": agree/n,
  "max_abs_dp": max(dps),
  "mean_ereshkigal_s": sum(r.get("total_seconds") or 0 for r in a)/n,
  "mean_semif_s": sum(r.get("total_seconds") or 0 for r in b)/n,
  "claim": "Measured same-GGUF llamacpp A/B",
}
Path("results/ab_llamacpp.json").write_text(json.dumps(rep, indent=2)+"\n")
print(json.dumps(rep, indent=2))
PY
else
  echo "SEMIF_SCORE unset; wrote $OUT_E only. Set SEMIF_SCORE to Python semif-score for a fair A/B."
  python3 - <<PY
import json
from pathlib import Path
a=[json.loads(l) for l in Path("$OUT_E").read_text().splitlines() if l.strip()]
n=len(a)
rep={"n":n,"argmax_agree": None, "note": "Python SemIf not run (SEMIF_SCORE unset)",
     "mean_ereshkigal_s": sum(r.get("total_seconds") or 0 for r in a)/max(n,1),
     "claim": "Proposed until SEMIF_SCORE is provided"}
Path("results/ab_llamacpp.json").write_text(json.dumps(rep, indent=2)+"\n")
print(json.dumps(rep, indent=2))
PY
fi

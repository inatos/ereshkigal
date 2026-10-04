#!/usr/bin/env bash
# Fair same-GGUF A/B: Ereshkigal vs Python SemIf llamacpp.
# Usage: ./scripts/ab_llamacpp.sh [gguf] [input.jsonl]
# Env: LLAMA_THREADS (default 8), N_GPU_LAYERS (0), BIN, SEMIF_SCORE, MODEL, REV
#      AB_MAX_DP (gate; default 1e-6 for 0.6B-style, set 1e-2 for Qwen3.5-4B)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GGUF="${1:-models/Qwen3-0.6B-Q8_0.gguf}"
INPUT="${2:-examples/decisions.jsonl}"
BIN="${BIN:-$ROOT/target/release/semif-score}"
if [[ ! -x "$BIN" ]]; then
  BIN="$ROOT/target/debug/semif-score"
fi
THREADS="${LLAMA_THREADS:-8}"
GPU="${N_GPU_LAYERS:-0}"
VENV_SCORE="$ROOT/.venv-semif/bin/semif-score"
if [[ -z "${SEMIF_SCORE:-}" && -x "$VENV_SCORE" ]]; then
  SEMIF_SCORE="$VENV_SCORE"
fi
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
REV="${REV:-c1899de289a04d12100db370d81485cdf75e47ca}"
# Heuristic: 4B tokenizer pin if the GGUF name looks like Qwen3.5-4B
if [[ "$GGUF" == *Qwen3.5-4B* || "$GGUF" == *qwen35* ]]; then
  MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
  REV="${REV:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
fi
if [[ "$GGUF" == *Qwen3.5-4B* ]]; then
  MODEL="Qwen/Qwen3.5-4B"
  REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
fi
mkdir -p results
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_E="results/ab_ereshkigal.jsonl"
rm -f "$OUT_E"
"$BIN" --mode direct --gguf "$GGUF" --input "$INPUT" --output "$OUT_E" \
  --llama-threads "$THREADS" --n-gpu-layers "$GPU" --no-replay \
  --model "$MODEL" --revision "$REV"

LLAMA2_VER=$(python3 - <<'PY'
import tomllib
from pathlib import Path
t=tomllib.loads(Path("Cargo.toml").read_text())
print(t["workspace"]["dependencies"]["llama-cpp-2"])
PY
)
GGUF_SHA=$(python3 - <<PY
import hashlib, pathlib
p=pathlib.Path("$GGUF")
h=hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1<<20), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)

if [[ -z "${SEMIF_SCORE:-}" ]]; then
  echo "SEMIF_SCORE unset and .venv-semif missing. Run ./scripts/setup_semif_venv.sh"
  python3 - <<PY
import json
from pathlib import Path
a=[json.loads(l) for l in Path("$OUT_E").read_text().splitlines() if l.strip()]
n=len(a)
rep={"n":n,"argmax_agree": None, "note": "Python SemIf not run (SEMIF_SCORE unset)",
     "mean_ereshkigal_s": sum(r.get("total_seconds") or 0 for r in a)/max(n,1),
     "gguf": "$GGUF", "gguf_sha256": "$GGUF_SHA",
     "llama_cpp_2": "$LLAMA2_VER", "threads": int("$THREADS"),
     "n_gpu_layers": int("$GPU"),
     "claim": "Proposed until SEMIF_SCORE is provided"}
Path("results/ab_llamacpp.json").write_text(json.dumps(rep, indent=2)+"\n")
print(json.dumps(rep, indent=2))
PY
  exit 1
fi

OUT_P="results/ab_semif_llamacpp.jsonl"
rm -f "$OUT_P"
"$SEMIF_SCORE" --mode direct --backend llamacpp --gguf "$GGUF" \
  --model "$MODEL" --revision "$REV" \
  --input "$INPUT" --output "$OUT_P" --llama-threads "$THREADS" --max-tokens 4096

GATE="${AB_MAX_DP:-1e-6}"
python3 - <<PY
import json, os
from pathlib import Path
a=[json.loads(l) for l in Path("$OUT_E").read_text().splitlines() if l.strip()]
b=[json.loads(l) for l in Path("$OUT_P").read_text().splitlines() if l.strip()]
by={r["id"]:r for r in b}
agree=0; dps=[]; sha_ok=0
for r in a:
    o=by[r["id"]]
    ai=max(range(len(r["probabilities"])), key=lambda i: r["probabilities"][i])
    bi=max(range(len(o["probabilities"])), key=lambda i: o["probabilities"][i])
    agree += ai==bi
    dps.append(max(abs(x-y) for x,y in zip(r["probabilities"], o["probabilities"])))
    sha_ok += r.get("prompt_sha256")==o.get("prompt_sha256")
n=len(a)
gate=float("$GATE")
max_dp=max(dps) if dps else 0.0
py_ver=None
try:
    import llama_cpp
    py_ver=llama_cpp.__version__
except Exception:
    py_ver="unknown"
rep={
  "n": n,
  "argmax_agree": agree/n,
  "argmax_agree_n": agree,
  "max_abs_dp": max_dp,
  "prompt_sha_agree": sha_ok/n,
  "mean_ereshkigal_s": sum(r.get("total_seconds") or 0 for r in a)/n,
  "mean_semif_s": sum(r.get("total_seconds") or 0 for r in b)/n,
  "gguf": "$GGUF",
  "gguf_sha256": "$GGUF_SHA",
  "model": "$MODEL",
  "revision": "$REV",
  "llama_cpp_2": "$LLAMA2_VER",
  "llama_cpp_python": py_ver,
  "threads": int("$THREADS"),
  "n_gpu_layers": int("$GPU"),
  "gate_max_abs_dp": gate,
  "stamp": "$STAMP",
  "claim": "Measured same-GGUF llamacpp A/B",
}
Path("results/ab_llamacpp.json").write_text(json.dumps(rep, indent=2)+"\n")
print(json.dumps(rep, indent=2))
if agree != n:
    raise SystemExit(f"argmax disagree {agree}/{n}")
if max_dp > gate:
    raise SystemExit(f"max |Δp|={max_dp} exceeds gate {gate}")
PY

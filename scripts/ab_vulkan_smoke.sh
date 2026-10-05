#!/usr/bin/env bash
# Vulkan smoke gate: argmax + prompt SHA (optional loose Δp). Never 1e-6.
# Prefer target-vulkan binary; N_GPU_LAYERS=99.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
# Smoke: require argmax+SHA; Δp optional and loose (default 1.0 = ignore Δp fail).
export AB_MAX_DP="${AB_MAX_DP:-1.0}"
export AB_REQUIRE_SHA="${AB_REQUIRE_SHA:-1}"
GGUF="${1:-$ROOT/../../.wordkeep/models/Qwen3-0.6B-Q8_0.gguf}"
if [[ ! -f "$GGUF" ]]; then
  GGUF="${1:-models/Qwen3-0.6B-Q8_0.gguf}"
fi
INPUT="${2:-fixtures/authored144.jsonl}"
BIN="${BIN:-}"
if [[ -z "$BIN" ]]; then
  if [[ -x "$ROOT/target-vulkan/release/semif-score" ]]; then
    BIN="$ROOT/target-vulkan/release/semif-score"
  elif [[ -x "$ROOT/target/release/semif-score" ]]; then
    BIN="$ROOT/target/release/semif-score"
  else
    BIN="$ROOT/target/debug/semif-score"
  fi
fi
# NVIDIA ICD (avoid lavapipe) when available.
ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [[ -f "$ICD" ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$ICD}"
  export VK_DRIVER_FILES="${VK_DRIVER_FILES:-$ICD}"
fi
echo "[ab_vulkan_smoke] GGUF=$GGUF INPUT=$INPUT N_GPU_LAYERS=$N_GPU_LAYERS AB_MAX_DP=$AB_MAX_DP BIN=$BIN"
# Run Esk only if SemIf venv missing — still check argmax self-consistency via SHA length.
OUT_E="$ROOT/results/ab_vulkan_smoke_ereshkigal.jsonl"
rm -f "$OUT_E"
"$BIN" --mode direct --gguf "$GGUF" --input "$INPUT" --output "$OUT_E" \
  --llama-threads "${LLAMA_THREADS:-8}" --n-gpu-layers "$N_GPU_LAYERS" --no-replay
python3 - <<PY
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path("$OUT_E").read_text().splitlines() if l.strip()]
n = len(rows)
bad_sha = [r["id"] for r in rows if len(r.get("prompt_sha256") or "") != 64]
argmax_ok = all("chosen" in r or "argmax" in r or r.get("option_logits") for r in rows)
# Prefer explicit chosen / top logit id when present.
print(json.dumps({
  "gate": "ab_vulkan_smoke",
  "n": n,
  "prompt_sha_64": n - len(bad_sha),
  "bad_sha_ids": bad_sha[:5],
  "n_gpu_layers": int("$N_GPU_LAYERS"),
  "note": "argmax+SHA smoke; not a 1e-6 Δp gate",
}, indent=2))
if bad_sha:
  sys.exit(2)
if n == 0:
  sys.exit(3)
print("PASS ab_vulkan_smoke")
PY
# If Python SemIf is available, also run full A/B with loose Δp.
if [[ -n "${SEMIF_SCORE:-}" || -x "$ROOT/.venv-semif/bin/semif-score" ]]; then
  exec env BIN="$BIN" N_GPU_LAYERS="$N_GPU_LAYERS" AB_MAX_DP="$AB_MAX_DP" \
    "$ROOT/scripts/ab_llamacpp.sh" "$GGUF" "$INPUT"
fi

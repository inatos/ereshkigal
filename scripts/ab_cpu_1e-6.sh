#!/usr/bin/env bash
# CPU ggml SoT gate: Ereshkigal vs Python SemIf @ 1e-6 (argmax + Δp).
# Never use Vulkan / n_gpu_layers>0 here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export N_GPU_LAYERS=0
export AB_MAX_DP="${AB_MAX_DP:-1e-6}"
GGUF="${1:-$ROOT/../../.wordkeep/models/Qwen3-0.6B-Q8_0.gguf}"
if [[ ! -f "$GGUF" ]]; then
  GGUF="${1:-models/Qwen3-0.6B-Q8_0.gguf}"
fi
INPUT="${2:-fixtures/authored144.jsonl}"
BIN="${BIN:-}"
if [[ -z "$BIN" ]]; then
  if [[ -x "$ROOT/target/release/semif-score" ]]; then
    BIN="$ROOT/target/release/semif-score"
  else
    BIN="$ROOT/target/debug/semif-score"
  fi
fi
echo "[ab_cpu_1e-6] GGUF=$GGUF INPUT=$INPUT N_GPU_LAYERS=0 AB_MAX_DP=$AB_MAX_DP"
exec env BIN="$BIN" N_GPU_LAYERS=0 AB_MAX_DP="$AB_MAX_DP" \
  "$ROOT/scripts/ab_llamacpp.sh" "$GGUF" "$INPUT"

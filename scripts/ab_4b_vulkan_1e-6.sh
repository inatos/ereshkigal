#!/usr/bin/env bash
# Measure bartowski 4B Vulkan at 1e-6 (argmax+SHA still required).
# Does NOT change ab_vulkan_smoke.sh default for 0.6B authored144.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
export AB_MAX_DP="${AB_MAX_DP:-1e-6}"
export AB_REQUIRE_SHA="${AB_REQUIRE_SHA:-1}"
GGUF="${1:-$ROOT/../../.wordkeep/models/bartowski-Qwen3.5-4B-Q4_K_M.gguf}"
INPUT="${2:-$ROOT/examples/decisions.jsonl}"
OUT_DIR="$ROOT/results/quality"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/ab_4b_vulkan_1e-6.json"
echo "[ab_4b_vulkan_1e-6] GGUF=$GGUF INPUT=$INPUT AB_MAX_DP=$AB_MAX_DP"
set +e
"$ROOT/scripts/ab_vulkan_smoke.sh" "$GGUF" "$INPUT"
rc=$?
set -e
python3 - <<PY
import json
from pathlib import Path
meta = {
  "gguf": "$GGUF",
  "input": "$INPUT",
  "ab_max_dp": "$AB_MAX_DP",
  "n_gpu_layers": int("$N_GPU_LAYERS"),
  "exit_code": $rc,
  "gate": "vulkan_4b_1e-6",
  "note": "Measurement only; 0.6B Vulkan smoke stays argmax+SHA with loose Δp",
  "status": "pass" if $rc == 0 else "measured_fail",
}
smoke = Path("$ROOT/results/ab_vulkan_smoke_ereshkigal.jsonl")
if smoke.is_file():
  rows = [json.loads(l) for l in smoke.read_text().splitlines() if l.strip()]
  meta["n_rows"] = len(rows)
Path("$REPORT").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY
exit 0

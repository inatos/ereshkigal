#!/usr/bin/env bash
# Measure bartowski 4B CPU ggml vs Python SemIf at 1e-6 (n=3 fixture).
# Does NOT flip 0.6B Vulkan smoke to 1e-6. Record max |Δp|; promote gate only if green.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export N_GPU_LAYERS=0
export AB_MAX_DP="${AB_MAX_DP:-1e-6}"
GGUF="${1:-$ROOT/../../.wordkeep/models/bartowski-Qwen3.5-4B-Q4_K_M.gguf}"
INPUT="${2:-$ROOT/examples/decisions.jsonl}"
if [[ ! -f "$INPUT" ]]; then
  INPUT="$ROOT/fixtures/expected_smoke.jsonl"
fi
# Prefer a tiny 3-row decisions file if present.
for cand in "$ROOT/examples/decisions.jsonl" "$ROOT/fixtures/lang/smoke3.jsonl"; do
  [[ -f "$cand" ]] || continue
  INPUT="$cand"
  break
done
OUT_DIR="$ROOT/results/quality"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/ab_4b_cpu_1e-6.json"
echo "[ab_4b_cpu_1e-6] GGUF=$GGUF INPUT=$INPUT AB_MAX_DP=$AB_MAX_DP"
set +e
"$ROOT/scripts/ab_cpu_1e-6.sh" "$GGUF" "$INPUT"
rc=$?
set -e
python3 - <<PY
import json, glob, os
from pathlib import Path
root = Path("$ROOT")
# Prefer newest ab_llamacpp JSON if present.
cands = sorted(root.glob("results/ab_llamacpp*.json"), key=lambda p: p.stat().st_mtime)
meta = {
  "gguf": "$GGUF",
  "input": "$INPUT",
  "ab_max_dp": "$AB_MAX_DP",
  "exit_code": $rc,
  "gate": "cpu_4b_1e-6",
  "note": "Promote to CI gate only if exit 0; else keep published 4B at 1e-2",
}
if cands:
  try:
    meta["latest_ab"] = json.loads(cands[-1].read_text())
  except Exception as e:
    meta["latest_ab_error"] = str(e)
meta["status"] = "pass" if $rc == 0 else "measured_fail"
Path("$REPORT").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2)[:2000])
PY
exit 0

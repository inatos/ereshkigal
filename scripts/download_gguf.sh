#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p models
REPO="${1:-unsloth/Qwen3.5-4B-GGUF}"
FILE="${2:-Qwen3.5-4B-Q4_K_M.gguf}"
OUT="models/${FILE}"
if [[ -f "$OUT" ]]; then
  echo "already present: $OUT"
  exit 0
fi
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" "$FILE" --local-dir models
elif python3 -c 'import huggingface_hub' 2>/dev/null; then
  python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id="$REPO", filename="$FILE")
os.makedirs("models", exist_ok=True)
shutil.copy2(path, "models/$FILE")
print("copied", "models/$FILE")
PY
else
  echo "Install huggingface_hub (pip) or the hf CLI" >&2
  exit 1
fi
sha256sum "$OUT" | tee "${OUT}.sha256"

#!/usr/bin/env bash
# Convert gated PEFT LoRA pack → GGUF adapter for llama.cpp / Wordkeep draft.
# Requires: student holdout BA >= 0.75 in packs/.../train_status.json (gate).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
PEFT="${PEFT:-$ROOT/packs/qwen3-0.6b-lora-dev}"
OUT="${OUT:-$REPO/.wordkeep/models/qwen3-0.6b-lora-dev.gguf}"
CONV_DIR="$ROOT/vendor/llama.cpp-convert"
PY="${PY:-$ROOT/training/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

if [[ ! -f "$PEFT/train_status.json" ]]; then
  echo "[convert] missing $PEFT/train_status.json" >&2
  exit 2
fi
GATE="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("gate_pass"))' "$PEFT/train_status.json")"
if [[ "$GATE" != "True" && "$GATE" != "true" ]]; then
  echo "[convert] gate_pass is $GATE — refuse convert until BA>=0.75" >&2
  exit 3
fi
if [[ ! -f "$CONV_DIR/convert_lora_to_gguf.py" ]]; then
  echo "[convert] missing vendored converter at $CONV_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
echo "[convert] PEFT=$PEFT → $OUT"
cd "$CONV_DIR"
"$PY" convert_lora_to_gguf.py \
  --outfile "$OUT" \
  --outtype f16 \
  --base-model-id Qwen/Qwen3-0.6B \
  --trust-remote-code \
  "$PEFT"
sha="$("$PY" -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb");
import itertools
[h.update(c) for c in iter(lambda:f.read(1<<20), b"")]; print(h.hexdigest())' "$OUT")"
bytes=$(stat -c%s "$OUT")
meta="$ROOT/results/quality/lora_gguf_convert.json"
"$PY" - <<PY
import json
from pathlib import Path
Path("$meta").write_text(json.dumps({
  "peft": "$PEFT",
  "outfile": "$OUT",
  "sha256": "$sha",
  "bytes": int("$bytes"),
  "converter_rev": Path("$CONV_DIR/LLAMA_CPP_REV.txt").read_text().strip() if Path("$CONV_DIR/LLAMA_CPP_REV.txt").is_file() else None,
  "gate_pass": True,
}, indent=2) + "\n")
print("wrote $meta")
print(json.dumps({"outfile":"$OUT","sha256":"$sha","bytes":int("$bytes")}, indent=2))
PY

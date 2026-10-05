#!/usr/bin/env bash
# 4B→0.6B LoRA distill pipeline scaffolding on fixtures/dev_gold.jsonl.
# Evidence bar: report BA/ECE on a held-out slice of *dev_gold* folds — never
# claim 1e-6 vs SemIf, never fit on authored144 test.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLD="${GOLD:-$ROOT/fixtures/dev_gold.jsonl}"
TEACHER_OUT="${TEACHER_OUT:-$ROOT/training/teacher_dev_gold.jsonl}"
LORA_OUT="${LORA_OUT:-$ROOT/packs/qwen3-0.6b-lora-dev}"
REPORT="${REPORT:-$ROOT/results/quality/lora_dev_gold_status.json}"
mkdir -p "$(dirname "$TEACHER_OUT")" "$(dirname "$REPORT")" "$LORA_OUT"

python3 - <<PY
import json
from pathlib import Path
gold = [json.loads(l) for l in Path("$GOLD").read_text().splitlines() if l.strip()]
assert all(r.get("split") == "dev" for r in gold)
# Fold stub: even ids = train proxy, odd = holdout for BA/ECE reporting.
hold = [r for i, r in enumerate(gold) if i % 3 == 0]
train = [r for i, r in enumerate(gold) if i % 3 != 0]
meta = {
  "gold": "$GOLD",
  "n": len(gold),
  "n_train_proxy": len(train),
  "n_holdout_proxy": len(hold),
  "evidence_bar": {
    "target_BA_without_verify": 0.75,
    "report": ["BA", "ECE"],
    "forbid": ["1e-6 vs SemIf", "fit on authored144 test"],
  },
  "status": "scaffolded",
  "note": "Run teacher_label.py then train_lora.py; fill BA/ECE after a real train.",
}
Path("$REPORT").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY

if [[ -n "${ERESHKIGAL_GGUF_VERIFY:-${ERESHKIGAL_GGUF:-}}" ]]; then
  GGUF="${ERESHKIGAL_GGUF_VERIFY:-$ERESHKIGAL_GGUF}"
  echo "[lora] teacher-labeling $GOLD with $GGUF"
  python3 "$ROOT/training/teacher_label.py" --in "$GOLD" --out "$TEACHER_OUT" --gguf "$GGUF" || {
    echo "[lora] teacher_label soft-failed; status remains scaffolded" >&2
  }
  python3 "$ROOT/training/train_lora.py" --teacher-jsonl "$TEACHER_OUT" --out "$LORA_OUT" --steps 0 || true
fi
echo "wrote $REPORT"

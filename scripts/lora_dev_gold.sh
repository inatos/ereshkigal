#!/usr/bin/env bash
# 4B→0.6B LoRA distill on fixtures/dev_gold.jsonl.
# Evidence bar: BA/ECE on frozen holdout — never 1e-6 vs SemIf, never authored144.
# Prefers 4B verify GGUF for teacher soft labels; falls back to 0.6B with a note.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLD="${GOLD:-$ROOT/fixtures/dev_gold.jsonl}"
TEACHER_OUT="${TEACHER_OUT:-$ROOT/training/teacher_dev_gold.jsonl}"
LORA_OUT="${LORA_OUT:-$ROOT/packs/qwen3-0.6b-lora-dev}"
REPORT="${REPORT:-$ROOT/results/quality/lora_dev_gold_status.json}"
# Prefer frozen holdout IDs (train grows when gold expands).
SPLIT_JSON="${SPLIT_JSON:-$ROOT/results/quality/lora_strict_split.json}"
STEPS="${STEPS:-1536}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}"
PROMPT_STYLE="${PROMPT_STYLE:-semif}"
mkdir -p "$(dirname "$TEACHER_OUT")" "$(dirname "$REPORT")" "$LORA_OUT"

PY=python3
if [[ -x "$ROOT/training/.venv/bin/python" ]]; then
  PY="$ROOT/training/.venv/bin/python"
fi

# Prefer 4B teacher; never silently prefer 0.6B when 4B exists.
GGUF=""
TEACHER_KIND=""
for cand in \
  "${ERESHKIGAL_GGUF_VERIFY:-}" \
  "$ROOT/../../.wordkeep/models/Qwen3.5-4B-Q4_K_M.gguf" \
  "$ROOT/../../.wordkeep/models/bartowski-Qwen3.5-4B-Q4_K_M.gguf"
do
  [[ -n "$cand" && -f "$cand" ]] || continue
  GGUF="$cand"
  TEACHER_KIND="4B"
  break
done
if [[ -z "$GGUF" ]]; then
  for cand in \
    "${ERESHKIGAL_GGUF:-}" \
    "$ROOT/../../.wordkeep/models/Qwen3-0.6B-Q8_0.gguf"
  do
    [[ -n "$cand" && -f "$cand" ]] || continue
    GGUF="$cand"
    TEACHER_KIND="0.6B"
    break
  done
fi

SCORE_BIN=""
for cand in \
  "$ROOT/target/release/semif-score" \
  "$ROOT/target-vulkan/release/semif-score" \
  "$(command -v semif-score || true)"
do
  [[ -n "$cand" && -x "$cand" ]] || continue
  SCORE_BIN="$cand"
  break
done

"$PY" - <<PY
import json
from pathlib import Path
gold = [json.loads(l) for l in Path("$GOLD").read_text().splitlines() if l.strip()]
assert all(r.get("split") == "dev" for r in gold)
assert len(gold) >= 30, f"expected >=30 gold rows, got {len(gold)}"
meta = {
  "gold": "$GOLD",
  "n": len(gold),
  "evidence_bar": {
    "target_BA_without_verify": 0.75,
    "report": ["BA", "ECE"],
    "forbid": ["1e-6 vs SemIf", "fit on authored144 test"],
  },
  "status": "running",
  "steps": int("$STEPS"),
  "lr": float("$LR"),
  "lora_r": int("$LORA_R"),
  "lora_alpha": int("$LORA_ALPHA"),
  "lora_targets": "$LORA_TARGETS",
  "teacher_kind": "$TEACHER_KIND" or None,
  "teacher_gguf": "$GGUF" or None,
}
Path("$REPORT").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY

if [[ -n "$GGUF" && -f "$GGUF" && -n "$SCORE_BIN" ]]; then
  echo "[lora] teacher-labeling $GOLD with $TEACHER_KIND ($GGUF) via $SCORE_BIN"
  # Prefer GPU layers for 4B when VRAM is free; caller may override N_GPU_LAYERS.
  if [[ "$TEACHER_KIND" == "4B" && -z "${N_GPU_LAYERS:-}" ]]; then
    export N_GPU_LAYERS=99
  fi
  if ! "$PY" "$ROOT/training/teacher_label.py" --in "$GOLD" --out "$TEACHER_OUT" --gguf "$GGUF" --bin "$SCORE_BIN"; then
    echo "[lora] teacher_label soft-failed" >&2
  fi
else
  echo "[lora] no GGUF or semif-score; skipping teacher_label" >&2
fi

"$PY" "$ROOT/training/train_lora.py" \
  --teacher-jsonl "$TEACHER_OUT" \
  --gold "$GOLD" \
  --out "$LORA_OUT" \
  --steps "$STEPS" \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-targets "$LORA_TARGETS" \
  --prompt-style "$PROMPT_STYLE" \
  --permute \
  --split-json "$SPLIT_JSON" \
  ${CPU:+--cpu}

"$PY" - <<PY
import json
from pathlib import Path
report = Path("$REPORT")
meta = json.loads(report.read_text()) if report.is_file() else {}
train_status = Path("$LORA_OUT/train_status.json")
split = Path("$SPLIT_JSON")
if split.is_file():
  meta["split"] = json.loads(split.read_text())
if train_status.is_file():
  ts = json.loads(train_status.read_text())
  meta["train"] = ts
  meta["status"] = ts.get("status", meta.get("status", "unknown"))
  for k in (
    "train_ids", "holdout_ids", "n_train", "n_holdout", "n_examples_aug",
    "unadapted_holdout", "teacher_holdout", "student_holdout",
    "baseline_holdout", "gate_pass", "blocked_reason", "holdout_note",
  ):
    if k in ts:
      meta[k] = ts[k]
  # Compat alias
  if "student_holdout" in ts and "baseline_holdout" not in ts:
    meta["baseline_holdout"] = ts.get("unadapted_holdout") or ts.get("teacher_holdout")
else:
  meta["status"] = "blocked"
  meta["blocked_reason"] = "train_status.json missing"
meta["note"] = "BA/ECE only on frozen holdout; never 1e-6; never authored144; gate BA>=0.75"
report.write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps({k: meta.get(k) for k in (
  "status","n","n_train","n_holdout","teacher_kind","unadapted_holdout",
  "teacher_holdout","student_holdout","gate_pass","blocked_reason"
)}, indent=2))
PY
echo "wrote $REPORT"

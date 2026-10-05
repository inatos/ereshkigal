#!/usr/bin/env bash
# Wording optimizer on fixtures/dev_gold.jsonl only (never authored144 test).
# Uses overlap-NLL proxy when no GGUF; with GGUF runs ereshkigal optimize on
# decrees that declare [[decree.variants]] + split=dev tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLD="${GOLD:-$ROOT/fixtures/dev_gold.jsonl}"
OUT="${OUT:-$ROOT/results/quality/dev_gold_optimize.json}"
mkdir -p "$(dirname "$OUT")"
python3 - <<PY
import json
from pathlib import Path
from collections import defaultdict

gold_path = Path("$GOLD")
rows = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
dev = [r for r in rows if r.get("split") == "dev"]
assert all(r.get("split") == "dev" for r in rows), "dev_gold must be split=dev only"
assert len(dev) >= 10, f"expected >=10 dev rows, got {len(dev)}"
assert all(len(r.get("options") or []) > 3 for r in dev), "all rows need >3 options"

def tokenize(s):
    return [t.lower() for t in "".join(c if c.isalnum() else " " for c in s).split() if len(t) > 1]

def overlap_nll(question, state, expect):
    q = tokenize(question)
    hay = tokenize(f"{state} {expect}")
    if not q:
        return 10.0
    hits = sum(1 for t in q if t in hay)
    p = max(1e-6, min(1 - 1e-6, hits / len(q)))
    import math
    return -math.log(p)

# Synthetic wording variants per family (identity + shortened + verbose).
by_fam = defaultdict(list)
for r in dev:
    by_fam[r["family"]].append(r)

report = {"gold": str(gold_path), "n_dev": len(dev), "families": {}, "note": "overlap proxy on dev_gold only; not authored144"}
for fam, items in sorted(by_fam.items()):
    base_q = items[0]["question"]
    variants = [
        ("identity", base_q),
        ("short", " ".join(base_q.split()[:6]) + "?"),
        ("verbose", base_q + " Choose the single best option from the list."),
    ]
    scores = []
    for vid, q in variants:
        acc = 0.0
        for r in items:
            expect = r["options"][r["label"]]["id"]
            acc += overlap_nll(q, r.get("state", ""), expect)
        scores.append({"id": vid, "mean_nll": acc / len(items)})
    winner = min(scores, key=lambda s: s["mean_nll"])["id"]
    report["families"][fam] = {"winner": winner, "nll": scores, "n": len(items)}

Path("$OUT").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print("wrote", "$OUT")
PY

# Optional: decree-library optimize when a Vulkan/CPU ereshkigal binary exists.
BIN="${ERESHKIGAL_BIN:-}"
for cand in \
  "$ROOT/target-vulkan/release/ereshkigal" \
  "$ROOT/target/release/ereshkigal" \
  "$BIN"
do
  [[ -n "$cand" && -x "$cand" ]] || continue
  LIB="${LIB:-$ROOT/decrees}"
  if [[ -d "$LIB" ]]; then
    echo "[optimize] trying decree optimize via $cand"
    "$cand" optimize --lib "$LIB" --decree std/support 2>/dev/null || true
  fi
  break
done

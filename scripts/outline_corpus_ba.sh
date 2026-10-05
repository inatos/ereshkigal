#!/usr/bin/env bash
# Score fixtures/outline_corpus.jsonl with direct-options-v1 and state-outline-v1.
# Quality-only BA/ECE columns — never authored144, never 1e-6.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLD="${GOLD:-$ROOT/fixtures/outline_corpus.jsonl}"
OUT_DIR="${OUT_DIR:-$ROOT/results/quality}"
REPORT="${REPORT:-$OUT_DIR/outline_corpus.json}"
GGUF="${ERESHKIGAL_GGUF:-$ROOT/../../.wordkeep/models/Qwen3-0.6B-Q8_0.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
BIN="${BIN:-}"
mkdir -p "$OUT_DIR"
if [[ -z "$BIN" ]]; then
  if [[ -x "$ROOT/target-vulkan/release/semif-score" && "${N_GPU_LAYERS}" != "0" ]]; then
    BIN="$ROOT/target-vulkan/release/semif-score"
  elif [[ -x "$ROOT/target/release/semif-score" ]]; then
    BIN="$ROOT/target/release/semif-score"
  else
    BIN="$ROOT/target/debug/semif-score"
  fi
fi
if [[ ! -f "$GGUF" ]]; then
  echo "[outline] missing GGUF=$GGUF — writing blocked report" >&2
  python3 -c "import json; from pathlib import Path; Path('$REPORT').write_text(json.dumps({'status':'blocked','reason':'missing GGUF','gguf':'$GGUF','gold':'$GOLD'}, indent=2)+'\n')"
  exit 0
fi
DIRECT_OUT="$OUT_DIR/outline_corpus_direct.jsonl"
OUTLINE_OUT="$OUT_DIR/outline_corpus_outline.jsonl"
echo "[outline] scoring direct-options-v1 via $BIN"
TOK_ARGS=()
case "$(basename "$GGUF")" in
  *0.6B*|*0_6B*)
    TOK_ARGS=(--model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca)
    ;;
esac
"$BIN" --mode direct --gguf "$GGUF" --input "$GOLD" --output "$DIRECT_OUT" \
  --prompt-version direct-options-v1 --n-gpu-layers "$N_GPU_LAYERS" --no-replay \
  "${TOK_ARGS[@]}"
echo "[outline] scoring state-outline-v1 via $BIN"
"$BIN" --mode direct --gguf "$GGUF" --input "$GOLD" --output "$OUTLINE_OUT" \
  --prompt-version state-outline-v1 --n-gpu-layers "$N_GPU_LAYERS" --no-replay \
  "${TOK_ARGS[@]}"
export GOLD DIRECT_OUT OUTLINE_OUT REPORT GGUF N_GPU_LAYERS
python3 <<'PY'
import json, os
from pathlib import Path

gold_path = Path(os.environ["GOLD"])
direct_path = Path(os.environ["DIRECT_OUT"])
outline_path = Path(os.environ["OUTLINE_OUT"])
report_path = Path(os.environ["REPORT"])

gold = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
by_id = {r["id"]: r for r in gold}

def load_preds(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = {}
    for r in rows:
        rid = r.get("id") or r.get("decision_id")
        if rid:
            out[rid] = r
    return out

def metrics(preds):
    y_true, y_pred, confs = [], [], []
    for gid, g in by_id.items():
        p = preds.get(gid)
        if not p:
            continue
        label = int(g["label"])
        opts = g["options"]
        chosen = None
        if isinstance(p.get("options"), list) and p["options"]:
            probs = [float(o.get("probability", 0.0)) for o in p["options"]]
            idx = max(range(len(probs)), key=lambda i: probs[i])
            chosen = idx
            confs.append(probs[idx])
        elif "chosen" in p or "option_ids" in p:
            cid = p.get("chosen")
            if cid is None and p.get("option_ids") and p.get("probabilities"):
                idx = max(range(len(p["probabilities"])), key=lambda i: float(p["probabilities"][i]))
                chosen = idx
                confs.append(float(p["probabilities"][idx]))
            else:
                chosen = next((i for i, o in enumerate(opts) if o["id"] == cid), None)
                probs = p.get("probabilities") or []
                confs.append(float(probs[chosen]) if chosen is not None and chosen < len(probs) else 0.0)
        if chosen is None:
            continue
        y_true.append(label)
        y_pred.append(chosen)
    n = len(y_true)
    if n == 0:
        return {"n": 0, "BA": None, "ECE": None}
    classes = sorted(set(y_true))
    recalls = []
    for c in classes:
        idx = [i for i, t in enumerate(y_true) if t == c]
        hits = sum(1 for i in idx if y_pred[i] == c)
        recalls.append(hits / len(idx))
    ba = sum(recalls) / len(recalls) if recalls else 0.0
    bins = 5
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            i
            for i, c in enumerate(confs)
            if (c >= lo and (c < hi or (b == bins - 1 and c <= hi)))
        ]
        if not members:
            continue
        acc = sum(1 for i in members if y_pred[i] == y_true[i]) / len(members)
        conf = sum(confs[i] for i in members) / len(members)
        ece += (len(members) / n) * abs(acc - conf)
    acc = sum(1 for i in range(n) if y_true[i] == y_pred[i]) / n
    return {"n": n, "BA": round(ba, 4), "ECE": round(ece, 4), "accuracy": round(acc, 4)}

report = {
    "status": "measured",
    "gold": str(gold_path),
    "n_gold": len(gold),
    "note": "Quality BA/ECE only; direct-options-v1 hashes unchanged; not authored144; not 1e-6",
    "direct_options_v1": metrics(load_preds(direct_path)),
    "state_outline_v1": metrics(load_preds(outline_path)),
    "gguf": os.environ["GGUF"],
    "n_gpu_layers": int(os.environ["N_GPU_LAYERS"]),
}
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
echo "wrote $REPORT"

#!/usr/bin/env bash
# Accuracy-first GGUF bakeoff on fixtures/authored144.jsonl (balanced accuracy).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BIN="${BIN:-$ROOT/target/release/semif-score}"
mkdir -p results models
if [[ ! -x "$BIN" ]]; then
  cargo build --release -p ereshkigal
  BIN="$ROOT/target/release/semif-score"
  export BIN
fi

GOLD="${GOLD:-fixtures/authored144.jsonl}"
SHARED="${SHARED:-fixtures/shared_state.jsonl}"
THREADS="${LLAMA_THREADS:-8}"
GPU_LAYERS="${N_GPU_LAYERS:-0}"

python3 - <<'PY'
import json, subprocess, hashlib, os, time
from pathlib import Path
from collections import defaultdict

manifest = json.loads(Path("manifests/default.json").read_text())
gold_path = Path(os.environ.get("GOLD", "fixtures/authored144.jsonl"))
shared_path = Path(os.environ.get("SHARED", "fixtures/shared_state.jsonl"))
bin_path = os.environ["BIN"]
threads = os.environ.get("THREADS", os.environ.get("LLAMA_THREADS", "8"))
gpu = os.environ.get("GPU_LAYERS", os.environ.get("N_GPU_LAYERS", "0"))

gold = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
gold_by_id = {r["id"]: r for r in gold}
n_classes = max(len(r["options"]) for r in gold)

def balanced_accuracy(pred_idx, gold_idx, n_classes):
    tp = [0.0] * n_classes
    support = [0.0] * n_classes
    for p, g in zip(pred_idx, gold_idx):
        if 0 <= g < n_classes:
            support[g] += 1.0
            if p == g:
                tp[g] += 1.0
    recalls = [tp[c] / support[c] for c in range(n_classes) if support[c] > 0]
    return sum(recalls) / len(recalls) if recalls else 0.0

report = [
    "# Ereshkigal GGUF bakeoff",
    "",
    f"Gold fixture: `{gold_path}` ({len(gold)} rows)",
    "Primary metric: **balanced accuracy** (mean per-class recall).",
    "Secondary: mean direct latency; serial/shared speedup on shared_state.",
    "",
]
best = None

for cand in manifest["bakeoff_candidates"]:
    repo, file = cand["repo"], cand["file"]
    local = Path("models") / file
    if not local.is_file():
        print(f"download {repo} {file}", flush=True)
        subprocess.check_call(["./scripts/download_gguf.sh", repo, file])
    out = Path("results") / f"bakeoff-{cand['id']}-authored144.jsonl"
    cmd = [
        bin_path, "--mode", "direct",
        "--gguf", str(local),
        "--input", str(gold_path),
        "--output", str(out),
        "--llama-threads", threads,
        "--n-gpu-layers", gpu,
    ]
    # Tokenizer source/revision from manifest default (Qwen3.5-4B pin)
    tok = manifest["tokenizer"]
    cmd += ["--model", tok["source"], "--revision", tok["revision"]]
    print("score", cand["id"], flush=True)
    t0 = time.time()
    subprocess.check_call(cmd)
    wall = time.time() - t0
    preds = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    pred_idx, gold_idx = [], []
    for p in preds:
        g = gold_by_id[p["id"]]
        probs = p["probabilities"]
        pred_idx.append(max(range(len(probs)), key=lambda i: probs[i]))
        gold_idx.append(int(g["label"]))
    ba = balanced_accuracy(pred_idx, gold_idx, n_classes)
    acc = sum(a == b for a, b in zip(pred_idx, gold_idx)) / len(pred_idx)
    mean_t = sum(p.get("total_seconds") or 0 for p in preds) / max(len(preds), 1)
    sha = hashlib.sha256(local.read_bytes()).hexdigest()

    # Secondary: shared_state timing
    serial_out = Path("results") / f"bakeoff-{cand['id']}-serial.jsonl"
    shared_out = Path("results") / f"bakeoff-{cand['id']}-shared.jsonl"
    serial_s = shared_s = None
    if shared_path.is_file():
        t0 = time.time()
        subprocess.check_call([
            bin_path, "--mode", "serial", "--gguf", str(local),
            "--input", str(shared_path), "--output", str(serial_out),
            "--llama-threads", threads, "--n-gpu-layers", gpu,
            "--model", tok["source"], "--revision", tok["revision"],
        ], stderr=subprocess.DEVNULL)
        serial_s = time.time() - t0
        t0 = time.time()
        subprocess.check_call([
            bin_path, "--mode", "shared", "--gguf", str(local),
            "--input", str(shared_path), "--output", str(shared_out),
            "--llama-threads", threads, "--n-gpu-layers", gpu,
            "--model", tok["source"], "--revision", tok["revision"],
        ], stderr=subprocess.DEVNULL)
        shared_s = time.time() - t0

    report.append(f"## {cand['id']}")
    report.append(f"- file: `{file}`")
    report.append(f"- sha256: `{sha}`")
    report.append(f"- balanced_accuracy: **{ba:.4f}**")
    report.append(f"- accuracy: {acc:.4f}")
    report.append(f"- mean total_s (direct): {mean_t:.3f}")
    report.append(f"- wall_s (authored144 direct): {wall:.1f}")
    if serial_s is not None:
        report.append(f"- shared_state serial_s: {serial_s:.3f}; shared_s: {shared_s:.3f}")
    report.append("")

    key = (ba, acc, -mean_t)
    if best is None or key > best[0]:
        best = (key, cand, sha, local.stat().st_size, ba)

if best:
    _, cand, sha, size, ba = best
    manifest["gguf"] = {
        "repo": cand["repo"],
        "file": cand["file"],
        "sha256": sha,
        "bytes": size,
        "notes": f"Bakeoff winner by balanced accuracy on authored144: {cand['id']} BA={ba:.4f}",
    }
    Path("manifests/default.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report.append(f"**Winner:** `{cand['id']}` (BA={ba:.4f}) → manifests/default.json updated.")

Path("results/bakeoff.md").write_text("\n".join(report) + "\n")
print("wrote results/bakeoff.md")
PY

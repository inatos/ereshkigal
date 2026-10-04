#!/usr/bin/env bash
# Score examples/decisions.jsonl with each manifests/default.json bakeoff candidate.
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

python3 - <<'PY'
import json, subprocess, hashlib, os
from pathlib import Path

manifest = json.loads(Path("manifests/default.json").read_text())
rows = Path("examples/decisions.jsonl").read_text().strip().splitlines()
report = ["# Ereshkigal GGUF bakeoff", "", f"Fixture rows: {len(rows)}", ""]
best = None
bin_path = os.environ["BIN"]
exp_path = Path("fixtures/expected_direct.jsonl")
exp = []
if exp_path.is_file():
    exp = [json.loads(l) for l in exp_path.read_text().splitlines() if l.strip()]
    exp_arg = [
        e["option_ids"][e["probabilities"].index(max(e["probabilities"]))] for e in exp
    ]
else:
    exp_arg = []

for cand in manifest["bakeoff_candidates"]:
    repo, file = cand["repo"], cand["file"]
    local = Path("models") / file
    if not local.is_file():
        print(f"download {repo} {file}")
        subprocess.check_call(["./scripts/download_gguf.sh", repo, file])
    out = Path("results") / f"bakeoff-{cand['id']}.jsonl"
    cmd = [
        bin_path,
        "--mode",
        "direct",
        "--gguf",
        str(local),
        "--input",
        "examples/decisions.jsonl",
        "--output",
        str(out),
        "--llama-threads",
        "8",
    ]
    print("score", cand["id"])
    subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
    preds = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    sha = hashlib.sha256(local.read_bytes()).hexdigest()
    argmaxes = [
        p["option_ids"][p["probabilities"].index(max(p["probabilities"]))] for p in preds
    ]
    mean_t = sum(p.get("total_seconds") or 0 for p in preds) / max(len(preds), 1)
    report.append(f"## {cand['id']}")
    report.append(f"- file: `{file}`")
    report.append(f"- sha256: `{sha}`")
    report.append(f"- argmax: {argmaxes}")
    report.append(f"- mean total_s: {mean_t:.3f}")
    agree = None
    if exp_arg:
        agree = sum(a == b for a, b in zip(argmaxes, exp_arg))
        report.append(f"- agree with expected_direct: {agree}/{len(exp_arg)}")
    report.append("")
    key = (agree if agree is not None else 0, -mean_t)
    if best is None or key > best[0]:
        best = (key, cand, sha, local.stat().st_size)

if best:
    _, cand, sha, size = best
    manifest["gguf"] = {
        "repo": cand["repo"],
        "file": cand["file"],
        "sha256": sha,
        "bytes": size,
        "notes": f"Bakeoff winner: {cand['id']}",
    }
    Path("manifests/default.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report.append(f"**Winner:** `{cand['id']}` → manifests/default.json updated.")
Path("results/bakeoff.md").write_text("\n".join(report) + "\n")
print("wrote results/bakeoff.md")
PY

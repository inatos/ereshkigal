#!/usr/bin/env bash
# Sole-MCP VRAM gate + sharp/uncertain cascade latency for tandem verify.
# Prefetch should warm draft (+ CPU tandem) only; 4B loads on first escalate.
# Evidence: cascade_source must be cascade-verify (not skipped) before claiming latency.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
OUT="${OUT:-$ROOT/results/quality/tandem_verify_bench.json}"
WK="${WK:-$REPO/tools/wordkeep}"
STATUS_DIR="${STATUS_DIR:-$REPO/.wordkeep}"
mkdir -p "$(dirname "$OUT")" "$STATUS_DIR"

refuse() {
  local reason="$1"
  python3 - "$OUT" "$reason" <<'PY'
import json, sys, time
from pathlib import Path
out, reason = sys.argv[1], sys.argv[2]
payload = {
  "status": "refused",
  "reason": reason,
  "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "note": "Do not claim cascade-verify latency when VRAM/MCP gate fails.",
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  exit 2
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  refuse "nvidia-smi missing"
fi
if ! nvidia-smi >/dev/null 2>&1; then
  refuse "nvidia driver unavailable (nvidia-smi failed)"
fi

FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"

mapfile -t WK_PIDS < <(pgrep -x wordkeep 2>/dev/null || true)
FILTERED=()
for pid in "${WK_PIDS[@]:-}"; do
  [[ -z "$pid" ]] && continue
  cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$cmd" == *wordkeep-wiki* ]] && continue
  FILTERED+=("$pid")
done
WK_PIDS=("${FILTERED[@]:-}")

echo "[tandem-bench] free_mib=$FREE_MIB used_mib=$USED_MIB existing_wordkeep=${WK_PIDS[*]:-none}"

if [[ ${#WK_PIDS[@]} -gt 1 ]]; then
  refuse "multiple wordkeep MCP processes (${WK_PIDS[*]}); keep at most one before verify claims"
fi

if [[ "${FREE_MIB:-0}" -lt 2500 && ${#WK_PIDS[@]} -eq 0 ]]; then
  refuse "free VRAM ${FREE_MIB} MiB < 2500 and no warm MCP; stop competing GPU holders"
fi

BIN="$WK/target/release/wordkeep"
if [[ ! -x "$BIN" ]]; then
  refuse "wordkeep release binary missing at $BIN — rebuild before bench"
fi

# One long-lived MCP stdio session: initialize → tools/call ×3 (sharp, uncertain, repeat).
python3 - "$BIN" "$REPO" "$OUT" "$FREE_MIB" "$USED_MIB" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path

bin_path, repo, out, free_mib, used_mib = sys.argv[1:6]

def decide(state, question, options):
    return {
        "state": state,
        "question": question,
        "options": [{"id": o, "description": o} for o in options],
    }

probes = [
    ("verify_first", decide(
        "A user cannot log in after a password reset email bounced.",
        "Which team should own this ticket?",
        ["identity", "payments", "growth", "data", "sre", "legal", "it", "other"],
    )),
    # Same undecided row again — should hit warm cascade-verify after lazy 4B load.
    ("verify_repeat", decide(
        "A user cannot log in after a password reset email bounced.",
        "Which team should own this ticket?",
        ["identity", "payments", "growth", "data", "sre", "legal", "it", "other"],
    )),
    ("draft_commit", decide(
        "Ambiguous multi-team incident with near-tie ownership signals across five routes.",
        "Pick the single best owner among these near-equally plausible options.",
        ["alpha", "bravo", "charlie", "delta", "echo"],
    )),
]

msgs = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "tandem-bench", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
]
for i, (label, args) in enumerate(probes, start=2):
    msgs.append({
        "jsonrpc": "2.0", "id": i, "method": "tools/call",
        "params": {"name": "semantic_decide", "arguments": args},
    })

stdin = "".join(json.dumps(m) + "\n" for m in msgs)
env = os.environ.copy()
env["WORDKEEP_ROOT"] = repo
t0 = time.perf_counter()
proc = subprocess.run(
    [bin_path, "--root", repo],
    input=stdin,
    text=True,
    capture_output=True,
    env=env,
    timeout=300,
)
wall_ms = int((time.perf_counter() - t0) * 1000)
Path("/tmp/tandem_bench_stderr.txt").write_text(proc.stderr or "")
Path("/tmp/tandem_bench_stdout.txt").write_text(proc.stdout or "")

# Map id -> result payload
by_id = {}
for line in (proc.stdout or "").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        o = json.loads(line)
    except Exception:
        continue
    if "id" not in o:
        continue
    r = o.get("result")
    payload = None
    if isinstance(r, dict):
        content = r.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = content[0].get("text")
            if text:
                if r.get("isError"):
                    payload = {"error": text}
                else:
                    try:
                        payload = json.loads(text)
                    except Exception:
                        payload = {"raw_text": text[:500]}
        elif "cascade_source" in r or "chosen" in r:
            payload = r
    by_id[o["id"]] = {"rpc": o, "payload": payload, "isError": bool(isinstance(r, dict) and r.get("isError"))}

results = {}
sources = []
for i, (label, _) in enumerate(probes, start=2):
    entry = by_id.get(i) or {}
    payload = entry.get("payload") or {}
    src = payload.get("cascade_source")
    if src is None and isinstance(payload.get("semif"), dict):
        src = payload["semif"].get("cascade_source")
    results[label] = {
        "cascade_source": src,
        "timing_us": payload.get("timing_us"),
        "chosen": payload.get("chosen"),
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
    }
    sources.append(src)

status_path = None
# wordkeep writes cache_dir()/semif-status.json (XDG cache), not .wordkeep/.
for cand in [
    Path.home() / ".cache" / "wordkeep" / "semif-status.json",
    Path(repo) / ".wordkeep" / "semif-status.json",
]:
    if cand.is_file():
        status_path = cand
        break
status = {}
if status_path and status_path.is_file():
    try:
        status = json.loads(status_path.read_text())
    except Exception:
        status = {}
status["status_path"] = str(status_path) if status_path else None

verify_hit = any(s == "cascade-verify" for s in sources)
adapter_path = status.get("adapter")
adapter_loaded = status.get("adapter_loaded")
payload = {
    "status": "ok" if verify_hit else ("rpc_error" if proc.returncode else "no_verify"),
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "free_mib_at_start": int(free_mib),
    "used_mib_at_start": int(used_mib),
    "wall_ms_session": wall_ms,
    "returncode": proc.returncode,
    "gguf_verify_loaded": status.get("gguf_verify_loaded"),
    "gguf_verify_lazy": status.get("gguf_verify_lazy"),
    "cold_load_ms": status.get("cold_load_ms"),
    "cpu_draft_loaded": status.get("cpu_draft_loaded"),
    "tandem": status.get("tandem"),
    "adapter": adapter_path,
    "adapter_loaded": adapter_loaded,
    "probes": results,
    "cascade_sources": sources,
    "stderr_tail": (proc.stderr or "")[-1500:],
    "note": (
        "cascade-verify observed; first uncertain pays cold 4B load under lazy verify"
        if verify_hit else
        "cascade-verify not observed — do not update latency tables; check sole MCP + lazy verify + VRAM"
    ),
    "adapter_note": (
        "draft LoRA attached (semif.adapter / ERESHKIGAL_ADAPTER); verify stays bare"
        if adapter_loaded else
        "no draft adapter loaded this session"
    ),
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if not verify_hit:
    sys.exit(3)
PY
echo "wrote $OUT"

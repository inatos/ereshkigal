#!/usr/bin/env bash
# Adapter shared-suffix vs direct microbench (n_seq_max=32).
# Records wall ms, per-row argmax agreement (must be 100%), and speedup.
# Sole MCP + free VRAM for GPU score.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
OUT="${OUT:-$ROOT/results/quality/adapter_shared_bench.json}"
IN="${IN:-$ROOT/fixtures/shared_state.jsonl}"
GGUF="${ERESHKIGAL_GGUF:-$REPO/.wordkeep/models/Qwen3-0.6B-Q8_0.gguf}"
ADAPTER="${ERESHKIGAL_ADAPTER:-$REPO/.wordkeep/models/qwen3-0.6b-lora-dev.gguf}"
SCALE="${ERESHKIGAL_ADAPTER_SCALE:-1.5}"
N_SEQ_MAX="${N_SEQ_MAX:-32}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
WARM_REPEATS="${WARM_REPEATS:-3}"
MODEL="${ERESHKIGAL_MODEL:-Qwen/Qwen3-0.6B}"
REVISION="${ERESHKIGAL_REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}"
mkdir -p "$(dirname "$OUT")"

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
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  exit 2
}

supports_adapter() {
  local bin="$1"
  "$bin" --help 2>&1 | grep -q -- '--adapter'
}

resolve_bin() {
  local c
  # Prefer locally built bins that know --adapter; ignore stale SEMIF_SCORE/venv stubs.
  for c in \
    "$ROOT/target/release/semif-score" \
    "$ROOT/target-vulkan/release/semif-score" \
    "${SEMIF_SCORE:-}" \
    "$(command -v semif-score || true)"; do
    [[ -n "$c" && -x "$c" ]] || continue
    if supports_adapter "$c"; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

BIN="$(resolve_bin || true)"
[[ -n "$BIN" ]] || refuse "semif-score binary missing"
[[ -f "$GGUF" ]] || refuse "GGUF missing: $GGUF"
[[ -f "$ADAPTER" ]] || refuse "adapter missing: $ADAPTER"
[[ -f "$IN" ]] || refuse "input missing: $IN"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  mapfile -t WK_PIDS < <(pgrep -x wordkeep 2>/dev/null || true)
  FILTERED=()
  for pid in "${WK_PIDS[@]:-}"; do
    [[ -z "$pid" ]] && continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$cmd" == *wordkeep-wiki* ]] && continue
    FILTERED+=("$pid")
  done
  WK_PIDS=("${FILTERED[@]:-}")
  if [[ ${#WK_PIDS[@]} -gt 1 ]]; then
    refuse "multiple wordkeep MCP processes (${WK_PIDS[*]}); keep at most one"
  fi
  if [[ ${#WK_PIDS[@]} -eq 1 && "${FREE_MIB:-0}" -lt 800 ]]; then
    refuse "warm MCP holds VRAM (free=${FREE_MIB} MiB); stop wordkeep for sole-GPU bench or free VRAM"
  fi
  if [[ ${#WK_PIDS[@]} -eq 0 && "${FREE_MIB:-0}" -lt 1500 ]]; then
    refuse "free VRAM ${FREE_MIB} MiB < 1500 for adapter shared bench"
  fi
else
  FREE_MIB=""
  N_GPU_LAYERS=0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

run_mode() {
  local mode="$1" out_jsonl="$2"
  # One cold + WARM_REPEATS warm; report cold separately, mean warm wall.
  local -a walls=()
  local i rc=0
  for i in $(seq 0 "$WARM_REPEATS"); do
    local t0 t1 ms
    t0="$(date +%s%N)"
    set +e
    "$BIN" --mode "$mode" \
      --gguf "$GGUF" \
      --adapter "$ADAPTER" \
      --adapter-scale "$SCALE" \
      --n-seq-max "$N_SEQ_MAX" \
      --n-gpu-layers "$N_GPU_LAYERS" \
      --model "$MODEL" \
      --revision "$REVISION" \
      --prompt-version direct-options-v1 \
      --input "$IN" \
      --output "$out_jsonl" \
      >/dev/null 2>"$TMP/${mode}_$i.log"
    rc=$?
    set -e
    t1="$(date +%s%N)"
    ms="$(python3 -c "print(round(($t1-$t0)/1e6))")"
    walls+=("$ms")
    [[ "$rc" -eq 0 ]] || return "$rc"
  done
  # Last log carries shared-batch total_s=… for score_sum when JSONL omits per-row seconds.
  if [[ -f "$TMP/${mode}_$WARM_REPEATS.log" ]]; then
    cp "$TMP/${mode}_$WARM_REPEATS.log" "$TMP/${mode}_last.log"
  fi
  printf '%s\n' "${walls[@]}"
}

echo "[adapter-shared] bin=$BIN gguf=$GGUF adapter=$ADAPTER scale=$SCALE n_seq_max=$N_SEQ_MAX gpu=$N_GPU_LAYERS"

DIRECT_WALLS="$(run_mode direct "$TMP/direct.jsonl")" || refuse "direct score failed (see $TMP/direct_*.log)"
SHARED_WALLS="$(run_mode shared "$TMP/shared.jsonl")" || refuse "shared score failed (see $TMP/shared_*.log)"

python3 - "$OUT" "$IN" "$TMP/direct.jsonl" "$TMP/shared.jsonl" "$DIRECT_WALLS" "$SHARED_WALLS" \
  "$GGUF" "$ADAPTER" "$SCALE" "$N_SEQ_MAX" "$N_GPU_LAYERS" "$BIN" "${FREE_MIB:-}" \
  "$TMP/direct_last.log" "$TMP/shared_last.log" <<'PY'
import json, sys, time
from pathlib import Path

(
    out,
    inp,
    direct_path,
    shared_path,
    direct_walls_s,
    shared_walls_s,
    gguf,
    adapter,
    scale,
    n_seq,
    n_gpu,
    bin_path,
    free_mib,
    direct_log,
    shared_log,
) = sys.argv[1:]

def load_jsonl(p):
    rows = []
    for line in Path(p).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def argmax_id(row):
    logits = row.get("option_logits")
    ids = row.get("option_ids") or []
    if isinstance(logits, dict):
        return max(logits, key=logits.get)
    if isinstance(logits, list) and ids:
        i = max(range(len(logits)), key=lambda j: logits[j])
        return ids[i]
    probs = row.get("probabilities") or []
    if probs and ids:
        i = max(range(len(probs)), key=lambda j: probs[j])
        return ids[i]
    return row.get("chosen")

def batch_total_ms(log_path):
    import re
    p = Path(log_path)
    if not p.is_file():
        return None
    text = p.read_text()
    m = re.search(r"total_s=([0-9.]+)", text)
    if not m:
        return None
    return round(float(m.group(1)) * 1000.0, 2)

def score_ms(rows, log_path):
    # Prefer engine-reported forward/total seconds (excludes process+GGUF load).
    tot = 0.0
    saw = False
    for r in rows:
        if "total_seconds" in r:
            tot += float(r["total_seconds"])
            saw = True
        elif "forward_seconds" in r:
            tot += float(r["forward_seconds"])
            saw = True
    if saw:
        return round(tot * 1000.0, 2)
    # Shared mode prints batch total_s=… on stderr instead of per-row seconds.
    return batch_total_ms(log_path)

direct = load_jsonl(direct_path)
shared = load_jsonl(shared_path)
assert len(direct) == len(shared) and direct, "empty or mismatched score outputs"
agree = sum(1 for d, s in zip(direct, shared) if argmax_id(d) == argmax_id(s))
n = len(direct)
agreement = agree / n

def parse_walls(s):
    return [int(x) for x in s.strip().splitlines() if x.strip()]

d_walls = parse_walls(direct_walls_s)
s_walls = parse_walls(shared_walls_s)
d_cold, d_warm = d_walls[0], d_walls[1:]
s_cold, s_warm = s_walls[0], s_walls[1:]
d_warm_mean = sum(d_warm) / len(d_warm) if d_warm else d_cold
s_warm_mean = sum(s_warm) / len(s_warm) if s_warm else s_cold
d_score = score_ms(direct, direct_log)
s_score = score_ms(shared, shared_log)
speedup_wall = (d_warm_mean / s_warm_mean) if s_warm_mean > 0 else None
speedup_score = (d_score / s_score) if s_score > 0 else None

payload = {
    "status": "ok" if agreement >= 1.0 - 1e-12 else "argmax_mismatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "n_rows": n,
    "n_seq_max": int(n_seq),
    "adapter_scale": float(scale),
    "n_gpu_layers": int(n_gpu),
    "free_mib": int(free_mib) if free_mib else None,
    "gguf": gguf,
    "adapter": adapter,
    "bin": bin_path,
    "input": inp,
    "argmax_agreement": agreement,
    "direct": {
        "cold_wall_ms": d_cold,
        "warm_wall_ms": d_warm,
        "warm_mean_wall_ms": round(d_warm_mean, 2),
        "score_sum_ms": d_score,
    },
    "shared": {
        "cold_wall_ms": s_cold,
        "warm_wall_ms": s_warm,
        "warm_mean_wall_ms": round(s_warm_mean, 2),
        "score_sum_ms": s_score,
    },
    "speedup_warm_wall_direct_over_shared": round(speedup_wall, 3) if speedup_wall is not None else None,
    "speedup_score_sum_direct_over_shared": round(speedup_score, 3) if speedup_score is not None else None,
    "note": "Process wall includes GGUF+adapter load each invoke; score_sum_ms is Σ total_seconds from JSONL (decode only). Prefer score_sum speedup for shared vs direct.",
}
if agreement < 1.0 - 1e-12:
    payload["mismatches"] = [
        {"i": i, "direct": argmax_id(d), "shared": argmax_id(s)}
        for i, (d, s) in enumerate(zip(direct, shared))
        if argmax_id(d) != argmax_id(s)
    ]
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if agreement < 1.0 - 1e-12:
    sys.exit(3)
PY

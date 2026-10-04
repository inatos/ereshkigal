# Ereshkigal GGUF bakeoff

Fixture rows: 3 (`examples/decisions.jsonl`)

## unsloth-qwen35-4b-q4km

- file: `Qwen3.5-4B-Q4_K_M.gguf`
- sha256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
- argmax: `["yes", "account_access", "not_required"]`
- mean total_s: ~3.0 (CPU, 8 threads)
- agree with expected_direct: 3/3

## bartowski-qwen35-4b-q4km

Not downloaded in the initial bakeoff (disk/time). Re-run `./scripts/bakeoff.sh` to challenge the pin.

**Winner:** `unsloth-qwen35-4b-q4km` → `manifests/default.json`.

# Ereshkigal GGUF bakeoff

Gold fixture: `fixtures/authored144.jsonl` (144 rows)
Primary metric: **balanced accuracy** (mean per-class recall).
Secondary: mean direct latency; serial/shared speedup on shared_state.

## unsloth-qwen35-4b-q4km
- file: `Qwen3.5-4B-Q4_K_M.gguf`
- sha256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
- balanced_accuracy: **0.8556**
- accuracy: 0.8542
- mean total_s (direct): 3.690
- shared_state serial_s: 18.802; shared_s: 17.146; speedup: 1.10x

## bartowski-qwen35-4b-q4km
- file: `Qwen_Qwen3.5-4B-Q4_K_M.gguf`
- sha256: `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`
- balanced_accuracy: **0.7870**
- accuracy: 0.7847
- mean total_s (direct): 3.375
- shared_state serial_s: 18.362; shared_s: 17.200; speedup: 1.07x

**Winner:** `unsloth-qwen35-4b-q4km` (BA=0.8556) → manifests/default.json updated.

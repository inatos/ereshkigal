# LoRA / probe training

Python 3.12 venv (host Python 3.14 has no CUDA torch wheels). Keep **torch
2.4.1+cu118** — measured Pascal `sm_61` on the 1080 Ti. Do **not** upgrade to
torch≥2.5 (CUDA disappears on Pascal).

```bash
cd training
uv venv .venv --python 3.12   # or sibling-link an existing .venv
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
python -c 'import torch, peft, transformers; print(torch.__version__, torch.cuda.is_available(), peft.__version__, transformers.__version__)'
```

Pinned stack (see `requirements.txt`): `transformers==4.51.3` (Qwen3 support),
`peft==0.13.2`, `accelerate==1.1.1`. Newer peft / transformers≥5 need torch≥2.5
and raise `NameError: nn` on 2.4.1.

**SemIf A/B stays on `../.venv-semif`** (or the sibling SemIf venv). Never reuse
`training/.venv` for 1e-6 gates or `semif-score` bakeoffs.

Student: Qwen3-0.6B. Teacher: soft labels from `teacher_label.py` on
`fixtures/dev_gold.jsonl` (prefer 4B verify GGUF when VRAM is free). Prompts are
**SemIf `direct-options-v1`** (Rust-parity) by default (`--prompt-style semif`).
Evidence bar is **BA/ECE on frozen holdout** — never 1e-6, never authored144.
Gate: student holdout **BA ≥ 0.75** before PEFT→GGUF conversion.
Frozen split: `results/quality/lora_strict_split.json` (train 52 / holdout 16 after
+7 holdout-only + train-only `mention` rows). Never train on holdout IDs. Post-gate
runtime knobs: `adapter_scale` (sweep winner **1.5**), `temperature` (OOF-fit;
applied **1.0**), `n_seq_max=32` (Wordkeep clamp ceiling 64). Reconvert GGUF when
PEFT weights change.

```bash
# Prefer sole MCP + free VRAM before CUDA train.
../scripts/lora_dev_gold.sh
# or:
python train_lora.py --teacher-jsonl teacher_dev_gold.jsonl \
  --gold ../fixtures/dev_gold.jsonl --out ../packs/qwen3-0.6b-lora-dev \
  --steps 1536 --lora-targets q_proj,k_proj,v_proj,o_proj \
  --split-json ../results/quality/lora_strict_split.json
# CPU smoke when CUDA is down:
python train_lora.py ... --steps 8 --cpu
```

## PEFT → GGUF

Vendored converter (llama.cpp commit matching `llama-cpp-sys-2 0.1.158`):

`tools/ereshkigal/vendor/llama.cpp-convert/`

```bash
# After gate_pass in packs/.../train_status.json:
../scripts/convert_lora_gguf.sh ../packs/qwen3-0.6b-lora-dev \
  ../../../.wordkeep/models/qwen3-0.6b-lora-dev.gguf
```

Wire the adapter into Wordkeep via `semif.adapter` or `ERESHKIGAL_ADAPTER`
(draft/CPU-draft only; 4B verify stays bare). Offline score:

```bash
semif-score --gguf .../Qwen3-0.6B-Q8_0.gguf \
  --adapter .../qwen3-0.6b-lora-dev.gguf --adapter-scale 1.5 \
  --n-seq-max 32 --input holdout.jsonl
# Shared vs direct microbench (argmax must agree 100%):
../scripts/adapter_shared_bench.sh
```

CI never installs torch. Local only.

## Continue from a gated pack

```bash
python train_lora.py --init-adapter ../packs/qwen3-0.6b-lora-dev \
  --out ../packs/qwen3-0.6b-lora-dev --gold ../fixtures/dev_gold.jsonl \
  --teacher-jsonl teacher_dev_gold.jsonl \
  --split-json ../results/quality/lora_strict_split.json \
  --steps 448 --lr 3e-5 --lora-targets q_proj,k_proj,v_proj,o_proj \
  --ce-weight 0.95 --kl-weight 0.05 --upsample-families mention --upsample-factor 4
../scripts/convert_lora_gguf.sh ../packs/qwen3-0.6b-lora-dev \
  ../../../.wordkeep/models/qwen3-0.6b-lora-dev.gguf
```


# LoRA / probe training

Python 3.12 venv (host Python 3.14 has no CUDA torch wheels):

```bash
cd training
source .venv/bin/activate   # created with: uv venv .venv --python 3.12
# Measured 2026-10-04: torch 2.4.1+cu118, cuda True, arch includes sm_60,
# GPU NVIDIA GeForce GTX 1080 Ti cc (6,1). Train FP32. CPU fallback if CUDA disappears.
```

Student: Qwen3-0.6B. Teacher: pinned 4B GGUF via `semif-score` JSONL (soft labels).
Augment by permuting option order. Convert adapters with llama.cpp `convert_lora_to_gguf.py`
from the llama-cpp-sys-2 0.1.158 vendor tree.

CI never installs torch. Local only.

```bash
python train_lora.py --help
```

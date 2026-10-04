# SemIf vs Ereshkigal (authored144)

Same labeled fixture: SemIf `benchmarks/data/authored144.jsonl` (MIT), 144 rows, 3 families × 48.

Two tables. Table A is the fair comparison: same-GGUF Python vs Rust. Table B is published SemIf BF16 vs Ereshkigal GGUF. Different weights. Do not headline B as a Rust win.

Fair A/B: `./scripts/setup_semif_venv.sh` then `./scripts/ab_llamacpp.sh`. SemIf pin recorded in the JSON (`vendor/SemIf-OpenJev` HEAD). Bindings: `llama-cpp-2 0.1.158` vs `llama-cpp-python 0.3.35`.

## A. Same-GGUF llamacpp A/B (Measured 2026-10-04)

CPU, 8 threads, `n_gpu_layers=0`, `prompt_sha256` agree 100%.

| GGUF | n | Argmax agree | max \|Δp\| | Gate | Esk s/row | SemIf s/row | Esk family BA | SemIf family BA |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 (3-row examples) | 3 | **3/3** | 2e-16 | 1e-6 | 1.48 | 1.24 | - | - |
| Qwen3-0.6B Q8_0 (authored144) | 144 | **144/144** | 2e-16 | 1e-6 | 0.503 | 0.490 | 0.520 | 0.520 |
| unsloth Qwen3.5-4B Q4_K_M (3-row, threads=1) | 3 | **3/3** | **7.87e-3** (`policy-1`) | 1e-2 | ~15 | ~14 | - | - |
| unsloth Qwen3.5-4B Q4_K_M (authored144, 8 threads) | 144 | **136/144** | **0.216** | 1e-2 (fails) | 3.61 | 3.59 | **0.854** | **0.830** |

**0.6B is bit-exact** on the full gold set. **4B is not a tight A/B:** same prompt SHA, same GGUF bytes, 8 argmax disagreements and max \|Δp\|=0.216. `threads=1` did not fix the 3-row 7.87e-3 gap. Treat 4B Python vs Rust as **two llama.cpp ABIs** (`llama-cpp-2 0.1.158` vs `llama-cpp-python 0.3.35` / Qwen3.5 hybrid KV), not as a scorer bug in the prompt. CI gates **0.6B only**. 4B quality claims should name the binding.

## B. Published BF16 vs GGUF (not same weights)

| System | Weights | Hardware | Accuracy | Global BA | Mean-family BA | Groups all-correct |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SemIf Python (published) | Qwen3.5-4B BF16 | RTX 3090 | - | - | **0.813** | - |
| Ereshkigal | unsloth Qwen3.5-4B Q4_K_M | CPU 8 th | 0.854 | 0.856 | **0.854** | 0.556 |
| Ereshkigal | bartowski Qwen3.5-4B Q4_K_M | CPU 8 th | 0.785 | 0.787 | 0.787 | 0.361 |
| SemIf Python (published) | Qwen3-0.6B BF16 | GPU | - | - | 0.440 | - |
| Ereshkigal | Qwen3-0.6B Q8_0 | CPU 8 th | 0.521 | 0.527 | 0.521 | 0.056 |
| Ereshkigal Vulkan | unsloth 4B Q4 | GTX 1080 Ti, 99 layers | 0.840 | - | **0.839** | 0.556 |

Unsloth Q4 beating BF16 **0.813** is a **checkpoint** difference. Bartowski Q4 (the GGUF SemIf documents) sits below BF16.

Family split (CPU unsloth 4B): evidence **0.930** · rule 0.863 · **candidate_selection 0.769**.

## Cascade (0.6B draft → unsloth 4B verify, offline join)

| Routing | Draft commit | Family BA | Groups all-correct |
| --- | ---: | ---: | ---: |
| Margin tau=0.35 | 93% | 0.529 | 0.056 |
| **Conformal \|C\|=1, α=0.1** | **14%** (q̂≈1.000) | **0.811** | 0.417 |
| 4B-only (CPU) | 0% | 0.854 | 0.556 |

0.6B softmax is overconfident (selective@80% ≈ base acc). Conformal singleton routing is the serving default; `--routing margin` remains a debug flag.

## candidate_selection levers (Zheng ICLR 2024)

PriDe (letter prior on 20% cyclic subset) **did not help**: 4B CS family BA 0.733 vs 0.769 baseline. Left **opt-in** (`--debias pride`).

Cyclic **permute** oracle (fixed letter→content map) on 12 CS rows: family BA **0.889 vs 0.556** baseline. Use `--debias permute` when quality > 3× forwards. Pairwise BT not needed on this slice.

## Latency

| Path | Number | Notes |
| --- | ---: | --- |
| SemIf 4B BF16 parallel (3090, published) | 20.03 dec/s | shape777, different bed |
| Ereshkigal 4B Q4 CPU | 3.69 s/row (~0.27 dec/s) | authored144 |
| **Ereshkigal 4B Q4 Vulkan 1080 Ti** | **0.447 s/row (2.24 dec/s)** | 99 layers; GPU ≠ CPU |Δp| |
| Ereshkigal 0.6B Q8 CPU | 0.50 s/row | same-GGUF A/B |
| shape15 0.6B Vulkan | 15 rows, direct sum 2.40 s | `N_GPU_LAYERS=99` |

CPU remains the bit-stable A/B. GPU numbers are latency-only.

## Other SemIf workloads

| Workload | SemIf 4B BF16 | Ereshkigal |
| --- | ---: | --- |
| WANLI 256 | 0.637 | **not vendored** (not in SemIf `benchmarks/data`) |
| TypeSafe 102 | 0.845 | **not vendored** |
| perturbations108 | 0.766 | downloaded via `./scripts/download_semif_fixtures.sh` (not scored this cut) |

## Calibration (unsloth authored144)

In-sample T≈1.116, ECE 0.061→0.058. OOF 5-fold ECE 0.061→**0.054**. SemIf published OOF 0.038 on BF16.

## How to reproduce the fair A/B

```bash
./scripts/setup_semif_venv.sh
.venv-semif/bin/pytest vendor/SemIf-OpenJev/tests/test_core.py \
  vendor/SemIf-OpenJev/tests/test_cli.py vendor/SemIf-OpenJev/tests/test_calibrate.py -q
cargo build --release -p ereshkigal
AB_MAX_DP=1e-6 ./scripts/ab_llamacpp.sh models/Qwen3-0.6B-Q8_0.gguf examples/decisions.jsonl
AB_MAX_DP=1e-2 ./scripts/ab_llamacpp.sh models/Qwen3.5-4B-Q4_K_M.gguf examples/decisions.jsonl
```

Vulkan: `./scripts/build_vulkan.sh` then `N_GPU_LAYERS=99` (needs `libvulkan.so`; vendored Khronos / SPIRV headers. No sudo if the ICD is installed).

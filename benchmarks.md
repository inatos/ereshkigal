# Ereshkigal benchmarks

Measured 2026-10-04 unless noted. Gold is SemIf MIT [`authored144`](https://github.com/TheoLeeCJ/SemIf-OpenJev) (144 rows, 3 families × 48) unless a row says otherwise.

Reproduce the fair A/B: `./scripts/setup_semif_venv.sh` then `./scripts/ab_llamacpp.sh`. Row-level JSON and caveats: [`results/comparison.md`](results/comparison.md).

## Hardware on purpose

SemIf's published latency (20 dec/s on Qwen3.5-4B BF16) was taken on an **RTX 3090**. GPU numbers here are a **GTX 1080 Ti** (Pascal, 11 GB, 2017) over **Vulkan**, not CUDA.

That is on purpose. Current economic conditions. I am not spending 3090 money to make a 4B decision scorer look fast. Used Pascal is still cheap. llama.cpp on this SKU has no CUDA path worth fighting, so we offload with Vulkan and keep **CPU** as the bit-stable A/B.

Do not divide 1080 Ti seconds into SemIf's 3090 RESULTS.md. Different silicon, different precision (GGUF Q4 vs BF16), different serving stack.

## Fair same-GGUF A/B (Python SemIf vs Ereshkigal)

CPU, 8 threads, `n_gpu_layers=0`. Prompt SHA agreed on every row. Bindings: `llama-cpp-2 0.1.158` vs `llama-cpp-python 0.3.35`.

| GGUF | n | Argmax | max \|Δp\| | Esk s/row | SemIf s/row | Esk family BA | SemIf family BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B Q8 (3-row examples) | 3 | **3/3** | ~0 | 0.61 | 0.55 | - | - |
| 0.6B Q8 authored144 | 144 | **144/144** | ~0 | 0.50 | 0.49 | **0.521** | **0.521** |
| 4B unsloth Q4 (3-row, 1 thread) | 3 | 3/3 | 0.008 | ~15 | ~14 | - | - |
| 4B unsloth Q4 authored144 | 144 | **136/144** | **0.216** | 3.61 | 3.59 | **0.854** | **0.830** |

- **0.6B is a real port.** Same file, same answers, same probabilities. CI gates this (max \|Δp\| &lt; 1e-6).
- **4B is two llama.cpp ABIs**, not two scorers. Same GGUF and prompt hash; 8 argmax flips. Hybrid KV / Qwen3.5. Name the binding when you quote 4B quality.

## Quality vs published SemIf BF16 (not the same weights)

| System | Weights | Acc | Family BA | Groups all-correct | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| SemIf published | 4B BF16, **RTX 3090** | - | 0.813 | - | 20.0 dec/s (shape777) |
| Ereshkigal CPU | unsloth 4B Q4 | 0.854 | **0.854** | 0.556 | 0.27 dec/s |
| Ereshkigal CPU | bartowski 4B Q4 | 0.785 | 0.787 | 0.361 | 0.30 dec/s |
| Ereshkigal Vulkan | unsloth 4B Q4, **GTX 1080 Ti** | 0.840 | 0.839 | 0.556 | **2.24 dec/s** |
| SemIf published | 0.6B BF16 | - | 0.440 | - | - |
| Ereshkigal CPU | 0.6B Q8 | 0.521 | 0.521 | 0.056 | 1.97 dec/s |

Unsloth 4B families (CPU): evidence **0.930** · rule 0.863 · **candidate_selection 0.769**.

Unsloth Q4 **0.854** vs published BF16 **0.813** is a **checkpoint** difference, not "Rust beats Python." Bartowski Q4 (0.787) is the quantized drop you would expect from the GGUF SemIf documents.

Vulkan 4B: **0.447 s/row** (p50 0.447, min 0.34, max 0.47) vs CPU **3.69 s/row** (~8×). GPU BA 0.839 is latency-only. Do not mix it into |Δp| gates.

shape15 (0.6B, Vulkan, 99 layers): 15-row direct wall sum **2.40 s**.

## Cascade (0.6B draft → unsloth 4B verify)

| Router | Draft commit | Family BA | Groups all-correct |
| --- | ---: | ---: | ---: |
| Margin τ=0.35 | 93% | 0.529 | 0.056 |
| Conformal \|C\|=1, α=0.1 | **14%** | **0.811** | 0.417 |
| 4B only | 0% | 0.854 | 0.556 |

0.6B softmax confidence is not a skip signal (selective@80% ≈ base accuracy). Conformal singleton routing is the default serving path.

## candidate_selection

| Method | Slice | Family BA vs baseline |
| --- | --- | --- |
| PriDe (Zheng et al., letter prior) | 48 CS, 4B | 0.733 vs 0.769 (**−3.6 pp**). Opt-in only. |
| Cyclic permute | 12 CS, 4B | **0.889 vs 0.556** (~3× forwards) |

## Calibration (unsloth authored144, CPU)

In-sample T≈1.116, ECE 0.061→0.058. OOF 5-fold ECE → **0.054**. SemIf published OOF 0.038 on BF16.

## Not scored here

WANLI-256 and TypeSafe-102 are not in SemIf `benchmarks/data` (not vendored). `perturbations108` can be fetched with `./scripts/download_semif_fixtures.sh`.

## Reproduce

```bash
./scripts/setup_semif_venv.sh
AB_MAX_DP=1e-6 ./scripts/ab_llamacpp.sh models/Qwen3-0.6B-Q8_0.gguf examples/decisions.jsonl
AB_MAX_DP=1e-2 ./scripts/ab_llamacpp.sh models/Qwen3.5-4B-Q4_K_M.gguf examples/decisions.jsonl

./scripts/build_vulkan.sh
N_GPU_LAYERS=99 ./scripts/bench_shape.sh models/Qwen3-0.6B-Q8_0.gguf
```

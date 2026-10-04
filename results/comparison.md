# SemIf vs Ereshkigal (authored144)

Same labeled fixture: SemIf `benchmarks/data/authored144.jsonl` (MIT), 144 rows, 3 families × 48.

**Not a same-weights A/B.** SemIf’s published quality numbers are **native BF16** direct logits on an RTX 3090. Ereshkigal scores **GGUF** on CPU (8 threads, `n_gpu_layers=0`). Quantization, GGUF publisher, and serving stack all differ.

SemIf source: [README Quality](https://github.com/TheoLeeCJ/SemIf-OpenJev#quality) / [docs/RESULTS.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/master/docs/RESULTS.md) (retrieved 2026-10-04).

## Authored decisions — balanced accuracy

| System | Weights | Hardware | Accuracy | Global BA | Mean-family BA |
| --- | --- | --- | ---: | ---: | ---: |
| SemIf Python (published) | Qwen3.5-4B BF16 | RTX 3090 | — | — | **0.813** |
| Ereshkigal `semif-score` | unsloth Qwen3.5-4B Q4_K_M | CPU 8 threads | 0.854 | 0.856 | **0.854** |
| Ereshkigal `semif-score` | bartowski Qwen3.5-4B Q4_K_M | CPU 8 threads | 0.785 | 0.787 | 0.787 |
| SemIf Python (published) | Qwen3-0.6B BF16 | GPU (native) | — | — | 0.440 |
| Ereshkigal `semif-score` | Qwen3-0.6B Q8_0 | CPU 8 threads | 0.521 | 0.527 | 0.521 |
| SemIf EXL3 bridge (published) | Qwen3.8-27B EXL3 5 bpw | GPU | — | — | **0.958** |

- **Global BA** = mean per-class recall over the pooled 144 rows (Ereshkigal bakeoff pin metric).
- **Mean-family BA** = unweighted mean of per-family BA (SemIf published metric). Families: `candidate_selection`, `evidence_interpretation`, `rule_application`.
- Unsloth Q4_K_M beating SemIf’s BF16 **0.813** is **not** “Rust is more accurate.” It is a different checkpoint. Bartowski Q4_K_M (the GGUF SemIf documents) sits **below** the BF16 baseline, which is the expected quantized direction.

## Calibration (authored144)

| System | T | Raw ECE | Calibrated ECE | Protocol |
| --- | ---: | ---: | ---: | --- |
| SemIf | 1.23 | 0.068 | **0.038** | out-of-fold (published) |
| Ereshkigal unsloth pin | 1.116 | 0.061 | 0.058 | in-sample fit on the same 144 rows |

Argmax is unchanged by T in both stacks. Ereshkigal’s ECE drop is small because the fit is in-sample and the 4B pin was already reasonably calibrated; SemIf’s 0.038 is the stronger claim (held-out folds). Report: `results/calibration/authored144.json`.

## Latency (not comparable as a bakeoff)

| Path | Number | Notes |
| --- | ---: | --- |
| SemIf direct vs compact JSON (same 4B, 21 criteria, 3090) | 1.023 s vs 5.332 s | **5.21×** generation/direct; 0 output tokens vs 111 |
| SemIf shape777 shared/parallel | 20.03 dec/s | 777 decisions, BF16 GPU; serial prefix 10.75 dec/s |
| Ereshkigal unsloth authored144 direct | 3.690 s/row mean | CPU, 144 independent prefills |
| Ereshkigal 0.6B Q8 authored144 direct | 0.508 s/row mean | CPU, CI smoke weights |
| Ereshkigal shared_state (3 rows) serial→shared | 18.8 s → 17.1 s | **1.10×**; fixture too small to show GPU-style reuse |

Do not divide SemIf 3090 numbers into Ereshkigal CPU seconds.

## Other SemIf workloads (not yet run in Ereshkigal)

| Workload | SemIf 4B BF16 | Ereshkigal |
| --- | ---: | --- |
| WANLI 256, BA | 0.637 | not scored |
| TypeSafe 102, equal-case agreement | 0.845 (Jev published 0.883) | not scored |
| Every judgment grid 36, acc | 0.806 | not scored |
| Perturbation BA (4B) | 0.766 | not scored |

## Prompt / serving parity (what *is* matched)

- `prompt_sha256` bit-exact vs HF tokenizer + SemIf `direct-options-v1` recipe (`enable_thinking` off).
- Direct / serial / shared argmax + probabilities within `1e-4` on owned `examples/decisions.jsonl` for the pinned 4B GGUF.
- Serial ≡ shared on `fixtures/shared_state.jsonl`.

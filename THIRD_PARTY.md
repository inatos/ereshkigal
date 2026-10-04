# Third-party notices

## SemIf / OpenJev

Ereshkigal reproduces the **interface pattern** and prompt/scoring approach of
[SemIf (formerly OpenJev)](https://github.com/TheoLeeCJ/SemIf-OpenJev)
(MIT License, Copyright (c) 2026 TheoLeeCJ).

Independent project; **not affiliated with or endorsed by** Jev, TypeSafe, or
TheoLeeCJ. Jev, TypeSafe, and other names and marks are the property of their
respective owners.

- Example decision fixtures under `examples/` follow SemIf’s public JSON schema (MIT).
- `fixtures/authored144.jsonl` is vendored from SemIf
  `benchmarks/data/authored144.jsonl` (MIT) for accuracy bakeoffs and calibration.
- `webgpu-demo/` is adapted from SemIf’s browser lab (MIT), including the vendored
  [wllama](https://github.com/ngxson/wllama) runtime under `webgpu-demo/vendor/wllama/`.

## llama.cpp / llama-cpp-2

Inference uses [llama.cpp](https://github.com/ggerganov/llama.cpp) via the
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) Rust bindings
(MIT OR Apache-2.0).

## Model weights

Default scoring targets open GGUF checkpoints derived from
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0).
CI smoke uses [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) GGUF
(Apache-2.0). Weights are **not** redistributed in this repository; see
`manifests/default.json`, `manifests/ci-smoke.json`, and `scripts/download_gguf.sh`.

Browser demo model URLs also reference MiniCPM5-2B GGUF (see `webgpu-demo/README.md`).

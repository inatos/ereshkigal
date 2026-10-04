# Third-party notices

## SemIf / OpenJev

Ereshkigal reproduces the **interface pattern** and prompt/scoring approach of
[SemIf (formerly OpenJev)](https://github.com/TheoLeeCJ/SemIf-OpenJev)
(MIT License, Copyright (c) 2026 TheoLeeCJ).

Independent project; **not affiliated with or endorsed by** Jev, TypeSafe, or
TheoLeeCJ. Jev, TypeSafe, and other names and marks are the property of their
respective owners.

Example decision fixtures under `examples/` and owned audit rows under
`fixtures/` follow SemIf’s public JSON schema (MIT).

## llama.cpp / llama-cpp-2

Inference uses [llama.cpp](https://github.com/ggerganov/llama.cpp) via the
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) Rust bindings
(MIT OR Apache-2.0).

## Model weights

Default scoring targets open GGUF checkpoints derived from
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0).
Weights are **not** redistributed in this repository; see
`manifests/default.json` and `scripts/download_gguf.sh`.

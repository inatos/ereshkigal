# Novel operators

Inspiration, not Betwixt-adopted Figments. Claims below are **Measured** in-repo or **Proposed** until a GPU/Python A/B exists.

## Letter-slot cascade (conformal default)

0.6B Q8 drafts option logits. **Default:** commit iff the split-conformal set has size 1 (`--routing conformal --alpha 0.1`); else a 4B GGUF verifies. `--routing margin --tau …` is debug-only (raw softmax margin).

```bash
semif-score cascade --input fixtures/authored144.jsonl \
  --draft results/bakeoff-qwen3-0.6b-q8-authored144.jsonl \
  --verify-predictions results/bakeoff-unsloth-qwen35-4b-q4km-authored144.jsonl \
  --output results/cascade_conformal.jsonl \
  --gold fixtures/authored144.jsonl --routing conformal --alpha 0.1
```

**Measured (2026-10-04):** margin `tau=0.35` skipped **93%** at family BA **0.529**. Conformal `|C|=1` skipped **14%** at family BA **0.811** (4B-only 0.854). 0.6B max-prob is not a skip signal.

## State outline (`state-outline-v1`)

Tree-sitter JSON walk (plus markdown/Python line heuristics) compresses `state` into a bounded outline, tagged `[state-outline-v1]`. **`direct-options-v1` hashes are unchanged.** Quality is a separate BA column; run `--prompt-version state-outline-v1`.

Wordkeep’s tree-sitter `outline` is the analog, not copied into this repo’s MCP.

## Radix prefix KV + replay cache

- **Radix:** token-prefix trie of llama.cpp `SeqState` (SGLang RadixAttention idea). Serial/shared prefills insert; longest prefix restores.
- **Replay:** exact `prompt_sha256` → last `ScoreResult` (Dream-RSI *prefix-only stored outcomes* metaphor; not an autonomous scheduler).

Disable with `--no-radix` / `--no-replay`.

## Parallel shared suffixes

`n_seq_max` (default 32) + `copy_kv_cache_seq` + one batched decode. This is the SemIf 20 dec/s mechanism; it needs GPU offload to show up. **Measured CPU** `fixtures/shape15.jsonl` (3 states × 5 criteria, 0.6B Q8): shared batches ~1.3 s wall each.

## GBNF letter check

`letter_gbnf(n)` emits `root ::= "A" | ...`. Before scoring, gathered slots are compared to **GGUF-tokenized prompt+letter boundary tokens** (`slots_match_letters(enc.slots, gguf_letter_ids)`). A self-comparison of `enc.slots` against itself is not a check.

## Decision language (2026-10-04)

Named decrees, `.esk` parser/formatter, program graphs, abstain, pairwise ranking, conformal sets, probes, LoRA pack loading, package manifests: see `docs/LANGUAGE.md`. `direct-options-v1` hashes unchanged (Measured: JS `webgpu-demo/prompt.mjs` matches `expected_direct.jsonl`).

## Figments / papers

| Source | Use |
| --- | --- |
| FIG-2026-006 Dream-RSI | replay cache only |
| FIG-2026-005 Spatiotemporal composability | batch/order/stop scoring jobs (`scripts/bench_shape.sh`) |
| FIG-2026-003 Attention | rejected; not architecture |
| SGLang RadixAttention | radix trie |
| Leviathan et al. speculative decoding | cascade on **logits** |
| Guo et al. temperature scaling | `fit_temperature_oof` |

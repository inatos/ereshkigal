# Novel operators

Inspiration, not Betwixt-adopted Figments. Claims below are **Measured** in-repo or **Proposed** until a GPU/Python A/B exists.

## Letter-slot cascade (Leviathan-style, no decode)

0.6B Q8 drafts option logits. If top-1 − top-2 softmax margin `> tau`, commit; else a 4B GGUF verifies.

```bash
semif-score --gguf models/Qwen3-0.6B-Q8_0.gguf --input fixtures/authored144.jsonl \
  --output /tmp/draft.jsonl --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
semif-score cascade --input fixtures/authored144.jsonl --draft /tmp/draft.jsonl \
  --output /tmp/cascade.jsonl --tau 0.8 --verify-gguf models/Qwen3.5-4B-Q4_K_M.gguf
```

**Measured (2026-10-04, CPU):** Qwen3-0.6B Q8_0 authored144 probability margins are heavy-tailed (p50 ≈ 0.998). At `tau=0.35` about **93%** of rows would skip verify. That is a latency win and a quality risk — 0.6B family BA is ~0.52 vs unsloth 4B ~0.85. Tune `tau` upward (0.9–0.99) if you care about BA.

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

`letter_gbnf(n)` emits `root ::= "A" | ...`. Slot IDs must match unique letter terminals before scoring.

## Figments / papers

| Source | Use |
| --- | --- |
| FIG-2026-006 Dream-RSI | replay cache only |
| FIG-2026-005 Spatiotemporal composability | batch/order/stop scoring jobs (`scripts/bench_shape.sh`) |
| FIG-2026-003 Attention | rejected; not architecture |
| SGLang RadixAttention | radix trie |
| Leviathan et al. speculative decoding | cascade on **logits** |
| Guo et al. temperature scaling | `fit_temperature_oof` |

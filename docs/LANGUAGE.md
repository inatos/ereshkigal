# Ereshkigal decision language

Two front ends, one IR:

- `.esk` (people) — see [GRAMMAR.md](GRAMMAR.md)
- TOML/JSON (agents) — JSON Schema via `ereshkigal schema`

`Library::load` accepts a directory of either. `ereshkigal convert` round-trips.

Identity: letter-logit readout only. `direct-options-v1` hashes are frozen. New recipes (`options-first-v1`, `state-outline-v1`, `probe-*-v1`) are versioned.

Runtime never decodes answer tokens. Abstain, guards, chaining, collections, pairwise sort, conformal sets, and expected-cost choice are defined in `ereshkigal-lang` and executed by `Runtime` in `ereshkigal-core`.

## CLI surface (language)

| Command | Status |
| --- | --- |
| `check` / `lint` / `fmt` / `convert` / `schema` | Real |
| `decide` / `run` | Real (needs `--gguf` or `ERESHKIGAL_GGUF`) |
| `test --lib decrees --gguf …` | Real scored gate; without `--gguf` only counts gold rows |
| `serve --stdio --lib … --gguf …` | Real JSON-RPC over stdio |
| `serve --http PORT --lib …` | Real JSON-RPC (`POST /rpc` or `POST /{method}`) |
| `lsp` | Diagnostics + completion + hover on `.esk` |
| `optimize` / `distill` | Wording pick on **split=dev**; logistic probe from JSONL |

Tokenizer is chosen from the GGUF filename (`0.6B` → Qwen3-0.6B pin; otherwise Qwen3.5-4B). Optional `N_GPU_LAYERS` for offload.

### Serve (stdio JSON-RPC)

```bash
./target/release/ereshkigal serve --stdio --lib decrees \
  --gguf models/Qwen3-0.6B-Q8_0.gguf
```

One JSON object per line:

| method | params | result |
| --- | --- | --- |
| `stats` | — | lib counts, `gguf_loaded`, recipe |
| `lint` | — | `{ok}` or error |
| `decide` | `decree`, `state` | [`Decided`](../ereshkigal-lang/src/decided.rs) |
| `run` | `program`, `state` | `ProgramResult` |
| `test` | `split` (default `all`) | `{n, correct, BA, ECE, …}` |

### LSP

```bash
./target/release/ereshkigal lsp
```

Full-document sync. On open/change of `.esk` (or `.toml`), publishes `textDocument/publishDiagnostics` for parse errors and library lint failures. Completion offers keywords + decree/program names. Hover shows decree question and options.

### Decree gold gate (0.6B)

```bash
./target/release/ereshkigal check decrees
./target/release/ereshkigal lint decrees
./target/release/ereshkigal test --lib decrees --split all \
  --gguf models/Qwen3-0.6B-Q8_0.gguf
```

Pass bar on this smoke GGUF: **n=6, BA=1.0** (std + examples). Do not 1e-6-gate Qwen3.5-4B.

Std program id: `release_gate`. Examples: `example_release_gate` (no merge clash when loading `decrees/`).

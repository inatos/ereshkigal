# Ereshkigal decision language

Two front ends, one IR:

- `.esk` (people) — see [GRAMMAR.md](GRAMMAR.md)
- TOML/JSON (agents) — JSON Schema via `ereshkigal schema`

`Library::load` accepts a directory of either. `ereshkigal convert` round-trips.

Identity: letter-logit readout only. `direct-options-v1` hashes are frozen. New recipes (`options-first-v1`, `state-outline-v1`, `probe-*-v1`) are versioned.

Runtime never decodes answer tokens. Abstain, guards, chaining, collections, pairwise sort, conformal sets, and expected-cost choice are defined in `ereshkigal-lang` and executed by `Runtime` in `ereshkigal-core`.

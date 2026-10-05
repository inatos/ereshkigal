# Python bindings (maturin)

Local-only wheel for `ereshkigal-py` (PyO3 abi3-py310). CI does **not** install
torch/maturin wheels.

```bash
cd tools/ereshkigal/ereshkigal-py
# Prefer a dedicated 3.12 venv (host 3.14 may lack wheels):
#   uv venv .venv --python 3.12 && source .venv/bin/activate
#   pip install maturin
maturin develop --release
python -c "import ereshkigal; print(ereshkigal.schema()[:80]); e=ereshkigal.Engine(); print(e.schema()[:40])"
# With a GGUF (loads llama.cpp; heavy):
#   e = ereshkigal.Engine(gguf="../../.wordkeep/models/Qwen3-0.6B-Q8_0.gguf")
#   print(e.decide("health ok", "Is it healthy?", [{"id":"yes","description":"Yes"},{"id":"no","description":"No"}]))
```

`Engine.schema()` / module `schema()` need no GGUF. `Engine.decide(...)` requires
`gguf=` and downloads the matching HF tokenizer on first use.

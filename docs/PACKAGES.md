# Packages

`Ereshkigal.toml` lists path / git / hf dependencies. `ereshkigal.lock` pins sha256s.

```
ereshkigal new mypkg
ereshkigal add std --path decrees/std
ereshkigal fetch
ereshkigal verify
ereshkigal publish --dry-run
```

Index: [registry/index.toml](../registry/index.toml). Adapters activate only when `base_gguf_sha256` matches the loaded GGUF. `add` should be followed by `ereshkigal test`.
Teacher labels: `python training/teacher_label.py --in fixtures/dev_gold.jsonl --out training/teacher_dev_gold.jsonl` (Vulkan: `N_GPU_LAYERS=99`).
Then `python training/train_lora.py --teacher-jsonl training/teacher_dev_gold.jsonl` (dev gold only — never authored144 test).
Wording optimizer: `./scripts/optimize_dev_gold.sh`. LoRA scaffold + BA/ECE bar: `./scripts/lora_dev_gold.sh`.

Vulkan headers (gitignored): `./scripts/vendor_vulkan_headers.sh` copies into `vendor/{vulkan-headers,spirv-prefix}`.

Dual A/B gates: `./scripts/ab_cpu_1e-6.sh` (CPU @ 1e-6) vs `./scripts/ab_vulkan_smoke.sh` (argmax+SHA).

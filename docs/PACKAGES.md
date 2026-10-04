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

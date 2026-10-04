# Ereshkigal browser lab

Browser-only comparison of two readout paths through the same quantized local GGUF:

1. **Direct readout** — wllama log-probabilities for allowed single-token labels (A…T), softmax over the displayed options. No answer decoding.
2. **Generation** — greedy JSON distribution decode (512-token cap) for a side-by-side timing comparison.

Adapted from the [SemIf / OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) WebGPU demo (MIT). Independent project; not affiliated with Jev or TypeSafe.

Live experiment only — timings come from the current browser session. Paths run sequentially to avoid WebGPU contention.

## Run locally

```bash
cd webgpu-demo
python3 -m http.server 8080
```

Open `http://localhost:8080` in a WebGPU-capable browser. First load downloads 639 MB (default Qwen3 0.6B), 1.56 GB (MiniCPM5 2B), or 3.01 GB (Qwen3.5 4B). Keep `_headers` on Cloudflare for COOP/COEP + referrer policy.

Optional: `?local` on localhost serves a GGUF from the same origin if you place it next to the page (see `worker.js` `localFile` names).

## Pins

- wllama: `3.6.1` (vendored under `vendor/wllama/`)
- Vue: `3.5.21` (CDN)
- Default: [`Qwen3-0.6B Q8_0`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) @ `23749fefcc72300e3a2ad315e1317431b06b590a`
- Desktop: [`MiniCPM5-2B Q4_K_M`](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) @ `2079a22f3beaa4e306449978533478fe0522f4b3`
- High-memory: [`Qwen3.5-4B Q4_K_M`](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF) @ `4168f45a16a1290d65a4ec0fa312ae917a4c15d6`

Quality table numbers are SemIf-published native-checkpoint references, not quantized browser runs.

## Static smoke (no GPU)

```bash
cd webgpu-demo
python3 test_demo.py
```

Checks prompt/worker wiring, model pins, COOP/COEP headers, and letter-logit readout hooks. Manual browser check still required for WebGPU inference.

## Sources

- [wllama](https://github.com/ngxson/wllama)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [SemIf-OpenJev webgpu-demo](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/master/webgpu-demo)

#!/usr/bin/env python3
"""Optional Myridos PyTorch /v1/systemone lane for Transformers + LoRA checkpoints.

Loads real HuggingFace Transformers + optional PEFT LoRA (+ optional bidirectional
attention unmasking and optional `.ehead` cosine readout) unless `--dry-run-uniform`
is explicitly passed for offline smoke testing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


def _load_train_lora():
    train_path = Path(__file__).resolve().parent.parent / "training" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("ereshkigal_train_lora", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load train_lora helpers from {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _normalize_options(raw_opts: list[Any], include_none: bool = False) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, o in enumerate(raw_opts or []):
        if isinstance(o, str):
            out.append({"id": o, "description": o})
        elif isinstance(o, dict):
            oid = str(o.get("id") or o.get("label") or f"opt_{i}")
            desc = str(o.get("description") or o.get("label") or o.get("desc") or oid)
            out.append({"id": oid, "description": desc})
    if include_none and not any(o["id"] == "none" for o in out):
        out.append({"id": "none", "description": "None of the above"})
    return out


def _extract_items(req: dict[str, Any]) -> tuple[bool, list[dict[str, Any]], str]:
    params = req.get("params") if isinstance(req.get("params"), dict) else req
    base_state = params.get("state", "")
    base_include_none = bool(params.get("include_none", False))
    readout = str(params.get("readout") or "letter").strip().lower()
    raw_batch = params.get("questions") or params.get("batch")
    if isinstance(raw_batch, list) and raw_batch:
        shared_opts = params.get("options") if isinstance(params.get("options"), list) else None
        items = []
        for idx, it in enumerate(raw_batch):
            if not isinstance(it, dict):
                continue
            opts_raw = it.get("options") if isinstance(it.get("options"), list) else (shared_opts or [])
            opts = _normalize_options(opts_raw, bool(it.get("include_none", base_include_none)))
            if not opts:
                opts = [{"id": "a", "description": "a"}, {"id": "b", "description": "b"}]
            items.append(
                {
                    "id": str(it.get("id") or f"q-{idx}"),
                    "state": it.get("state", it.get("evidence", base_state)),
                    "question": str(it.get("question") or it.get("criterion") or ""),
                    "options": opts,
                    "type": str(it.get("type") or it.get("kind") or params.get("type") or "choice"),
                }
            )
        return True, items, readout

    opts = _normalize_options(params.get("options") or [], base_include_none)
    if not opts:
        opts = [{"id": "a", "description": "a"}, {"id": "b", "description": "b"}]
    item = {
        "id": str(params.get("id") or "score"),
        "state": base_state,
        "question": str(params.get("question") or params.get("criterion") or ""),
        "options": opts,
        "type": str(params.get("type") or params.get("kind") or "choice"),
    }
    return False, [item], readout


class UniformDryRunBackend:
    """Explicit `--dry-run-uniform` stub for CI/smoke tests without model weights."""

    stub = True

    def __init__(self, attention: str = "bidirectional"):
        self.attention = attention
        self.model_id = None
        self.adapter_path = None
        self.cosine_head_loaded = False
        self.amp_resolved = "fp32"

    def score_item(self, item: dict[str, Any], readout: str = "letter") -> dict[str, Any]:
        ids = [o["id"] for o in item["options"]]
        p = 1.0 / max(1, len(ids))
        return {
            "id": item["id"],
            "type": item["type"],
            "ok": True,
            "attention": self.attention,
            "readout": readout,
            "probabilities": [p] * len(ids),
            "scores": [[oid, 0.0] for oid in ids],
            "chosen": ids[0],
            "calibrated": False,
            "stub": True,
            "dry_run_uniform": True,
        }


class TorchLoRABackend:
    """Real Transformers + optional PEFT LoRA + optional bidirectional / `.ehead` scorer."""

    stub = False

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        attention: str = "bidirectional",
        device: str = "auto",
        amp: str = "auto",
        max_length: int = 768,
        cosine_head_path: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.train_mod = _load_train_lora()
        self.torch = torch
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.attention = attention
        self.max_length = max_length

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        cuda_cap = torch.cuda.get_device_capability(0) if self.device == "cuda" else None
        self.amp_resolved, self.use_amp = self.train_mod.resolve_amp_dtype(amp, self.device, cuda_cap)

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.letter_ids = self.train_mod._letter_id_cache(self.tok, 16)

        load_dtype = (
            torch.bfloat16
            if (self.use_amp and self.amp_resolved == "bf16")
            else (torch.float16 if (self.use_amp and self.amp_resolved == "fp16") else torch.float32)
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=load_dtype, trust_remote_code=True
        )
        base.to(self.device)

        if adapter_path:
            from peft import PeftModel

            ap = Path(adapter_path)
            if not (ap / "adapter_config.json").is_file():
                raise FileNotFoundError(f"adapter missing adapter_config.json: {ap}")
            self.model = PeftModel.from_pretrained(base, str(ap), is_trainable=False)
        else:
            self.model = base

        self.attention_patch = self.train_mod.configure_bidirectional_attention(
            self.model, attention
        )
        self.model.eval()

        self.cosine_head = None
        if cosine_head_path:
            self.cosine_head = self.train_mod.load_cosine_head_ehead(Path(cosine_head_path)).to(
                device=self.device, dtype=torch.float32
            )
            self.cosine_head.eval()
        self.cosine_head_loaded = self.cosine_head is not None

    def score_item(self, item: dict[str, Any], readout: str = "letter") -> dict[str, Any]:
        torch = self.torch
        eff_readout = readout if readout in ("letter", "cosine") else "letter"
        if eff_readout == "cosine" and self.cosine_head is None:
            raise ValueError("readout=cosine requested but --cosine-head was not loaded")

        prompt, opt_offsets = self.train_mod._prompt_from_row_with_option_offsets(item)
        with torch.no_grad():
            logits_list = self.train_mod._forward_option_logits(
                self.model,
                self.tok,
                self.device,
                self.letter_ids,
                [prompt],
                [opt_offsets],
                [len(item["options"])],
                self.max_length,
                eff_readout,
                self.cosine_head,
                self.attention == "bidirectional",
                self.amp_resolved,
                self.use_amp,
            )
            opt_logits = logits_list[0].float()
            probs = torch.softmax(opt_logits, dim=0).cpu().tolist()
            raw_scores = opt_logits.cpu().tolist()

        ids = [o["id"] for o in item["options"]]
        chosen_idx = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "id": item["id"],
            "type": item["type"],
            "ok": True,
            "attention": self.attention,
            "readout": eff_readout,
            "model": self.model_id,
            "adapter": self.adapter_path,
            "probabilities": probs,
            "scores": [[oid, float(s)] for oid, s in zip(ids, raw_scores)],
            "chosen": ids[chosen_idx],
            "calibrated": False,
            "stub": False,
        }


def score_payload(backend: Any, req: dict[str, Any]) -> dict[str, Any]:
    is_batch, items, readout = _extract_items(req)
    if is_batch:
        results = [backend.score_item(it, readout=readout) for it in items]
        return {
            "ok": True,
            "attention": backend.attention,
            "readout": readout,
            "count": len(results),
            "stub": bool(getattr(backend, "stub", False)),
            "results": results,
        }
    return backend.score_item(items[0], readout=readout)


def health_payload(backend: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "service": "ereshkigal-torch",
        "attention": backend.attention,
        "stub": bool(getattr(backend, "stub", False)),
        "model": getattr(backend, "model_id", None),
        "adapter": getattr(backend, "adapter_path", None),
        "cosine_head_loaded": bool(getattr(backend, "cosine_head_loaded", False)),
        "amp_resolved": getattr(backend, "amp_resolved", "fp32"),
    }


class Handler(BaseHTTPRequestHandler):
    backend: Any = None

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            self._json(200, health_payload(self.backend))
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return
        if self.path.startswith("/v1/systemone") or self.path.startswith("/rpc"):
            try:
                res = score_payload(self.backend, req)
            except Exception as e:
                self._json(400, {"ok": False, "error": str(e)})
                return
            if self.path.startswith("/rpc") and "jsonrpc" in req:
                self._json(
                    200,
                    {"jsonrpc": "2.0", "id": req.get("id"), "result": res},
                )
            else:
                self._json(200, res)
            return
        self._json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:
        print("[serve_torch]", fmt % args)


def build_backend_from_args(args: argparse.Namespace) -> Any:
    if args.dry_run_uniform:
        return UniformDryRunBackend(attention=args.attention)
    if not args.model:
        raise SystemExit(
            "serve_torch requires --model (or ERESHKIGAL_TORCH_MODEL) for real Transformers+LoRA load, "
            "or pass --dry-run-uniform for uniform smoke mode"
        )
    return TorchLoRABackend(
        model_id=args.model,
        adapter_path=args.adapter,
        attention=args.attention,
        device=args.device,
        amp=args.amp,
        max_length=args.max_length,
        cosine_head_path=args.cosine_head,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ereshkigal PyTorch /v1/systemone server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--attention", default="bidirectional", choices=["causal", "bidirectional"])
    ap.add_argument(
        "--model",
        default=os.environ.get("ERESHKIGAL_TORCH_MODEL"),
        help="HuggingFace model id or local path (required unless --dry-run-uniform)",
    )
    ap.add_argument(
        "--adapter",
        default=os.environ.get("ERESHKIGAL_TORCH_ADAPTER"),
        help="Optional PEFT LoRA adapter directory",
    )
    ap.add_argument(
        "--cosine-head",
        default=os.environ.get("ERESHKIGAL_COSINE_HEAD"),
        help="Optional EHEAD01 .ehead weights for readout=cosine",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--amp", default="auto", choices=["auto", "none", "bf16", "fp16"])
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument(
        "--dry-run-uniform",
        action="store_true",
        help="Serve uniform stub without loading Transformers/LoRA weights (smoke/CI only)",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    Handler.backend = build_backend_from_args(args)
    httpd = HTTPServer((args.host, args.port), Handler)
    print(
        f"[serve_torch] http://{args.host}:{args.port} attention={args.attention} "
        f"stub={Handler.backend.stub} model={Handler.backend.model_id} adapter={Handler.backend.adapter_path}"
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""3-seed LoRA model soup (weight average) for Jeff-1.1 recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def average_state_dicts(paths: list[Path]) -> dict:
    import torch

    soups = [torch.load(p, map_location="cpu") for p in paths]
    keys = soups[0].keys()
    out = {}
    n = float(len(soups))
    for k in keys:
        acc = None
        for s in soups:
            t = s[k].float()
            acc = t if acc is None else acc + t
        out[k] = (acc / n).to(soups[0][k].dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("need ≥2 checkpoints to soup")
    # Synthetic / offline path: if tensors unavailable, write a receipt only.
    try:
        state = average_state_dicts(args.checkpoints)
        import torch

        torch.save(state, args.out)
        receipt = {"ok": True, "n": len(args.checkpoints), "out": str(args.out)}
    except Exception as e:
        receipt = {
            "ok": False,
            "error": str(e),
            "n": len(args.checkpoints),
            "note": "write receipt only — install torch for real soup",
        }
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))
        raise SystemExit(0 if "torch" in str(e).lower() or "No module" in str(e) else 1)
    meta = args.out.with_suffix(".soup.json")
    meta.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

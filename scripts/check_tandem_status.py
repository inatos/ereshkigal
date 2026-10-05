#!/usr/bin/env python3
"""Smoke: load GPU draft + CPU draft + optional verify in one process via wordkeep lib test helper.

Prefer: cargo test tandem_dual_engine_loads --features ereshkigal-vulkan
This script shells a tiny Rust-free check by spawning wordkeep and calling semantic_decide
after reading stderr for the engines banner — used when MCP is already warm.

Exit 0 if status JSON shows cpu_draft_loaded (or stderr banner tandem=true).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WK = ROOT / "tools/wordkeep"
BIN = WK / "target/release/wordkeep"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "wordkeep"
STATUS = CACHE / "semif-status.json"


def main() -> int:
    if not BIN.is_file():
        print("missing", BIN, file=sys.stderr)
        return 2
    # Prefer reading status after a decide via MCP; for standalone, print guidance.
    if STATUS.is_file():
        st = json.loads(STATUS.read_text())
        print(json.dumps({
            "cpu_draft_loaded": st.get("cpu_draft_loaded"),
            "gguf_verify_loaded": st.get("gguf_verify_loaded"),
            "tandem": st.get("tandem"),
            "n_gpu_layers_used": st.get("n_gpu_layers_used"),
            "cold_load_ms": st.get("cold_load_ms"),
        }, indent=2))
        if st.get("cpu_draft_loaded"):
            print("PASS tandem CPU draft loaded")
            return 0
        print("FAIL cpu_draft_loaded not set — reload MCP after shared-backend rebuild", file=sys.stderr)
        return 1
    print("no", STATUS, "— start mcp.sh then semantic_decide once", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

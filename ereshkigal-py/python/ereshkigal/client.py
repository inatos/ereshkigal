"""JSON-RPC client for `ereshkigal serve`."""
from __future__ import annotations
import json
import subprocess
from typing import Any, Optional

class Client:
    def __init__(self, cmd: Optional[list[str]] = None):
        self.cmd = cmd or ["ereshkigal", "serve", "--stdio", "--lib", "decrees"]
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self._n = 0

    def call(self, method: str, params: Any = None) -> Any:
        self._n += 1
        req = {"jsonrpc": "2.0", "id": self._n, "method": method, "params": params or {}}
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line)

    def close(self):
        self.proc.terminate()

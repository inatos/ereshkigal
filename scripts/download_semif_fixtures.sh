#!/usr/bin/env bash
# Optional SemIf MIT extras (perturbations108). WANLI/TypeSafe are not in
# benchmarks/data — follow SemIf docs/REPRODUCE.md rather than vendoring HF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/fixtures"
base=https://raw.githubusercontent.com/TheoLeeCJ/SemIf-OpenJev/master/benchmarks/data
curl -fsSL "$base/perturbations108.jsonl" -o "$ROOT/fixtures/perturbations108.jsonl"
echo "wrote fixtures/perturbations108.jsonl"
echo "WANLI-256 / TypeSafe-102: not redistributed here (build from SemIf reproduce docs)."

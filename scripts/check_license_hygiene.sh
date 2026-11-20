#!/usr/bin/env bash
# Fail if tracked tools/ereshkigal trees leak CC BY-NC / non-commercial pins.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAIL=0
PATTERN='CC[[:space:]-]?BY[[:space:]-]?NC|BY-NC|non-?commercial|creativecommons.org/licenses/by-nc'

while IFS= read -r -d '' f; do
  if grep -nEi "$PATTERN" "$f" >/dev/null 2>&1; then
    echo "LICENSE LEAK: $f"
    grep -nEi "$PATTERN" "$f" | head -5
    FAIL=1
  fi
done < <(find "$ROOT/manifests" "$ROOT/decrees" "$ROOT/packs" -type f \( -name '*.json' -o -name '*.toml' -o -name '*.md' -o -name '*.esk' \) -print0 2>/dev/null)

# Negative self-test when CHECK_LICENSE_INJECT=1
if [[ "${CHECK_LICENSE_INJECT:-0}" == "1" ]]; then
  tmp="$(mktemp)"
  echo 'license: CC-BY-NC-4.0' >"$tmp"
  if ! grep -nEi "$PATTERN" "$tmp" >/dev/null; then
    echo "negative test failed: injector not detected"
    FAIL=1
  else
    echo "negative inject detected OK"
  fi
  rm -f "$tmp"
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "check_license_hygiene: FAIL (move NC pins to .wordkeep/models/manifest.local.json)" >&2
  exit 1
fi
echo "check_license_hygiene: OK"

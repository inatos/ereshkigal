#!/usr/bin/env bash
# Populate tools/ereshkigal/vendor/{vulkan-headers,spirv-prefix} from a sibling
# checkout or explicit env paths. Headers are gitignored; Betwixt builds prefer
# this in-tree tree so cmake does not require ~/Documents/_Projects/ereshkigal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor"
SRC="${ERESHKIGAL_VENDOR_SRC:-}"
if [[ -z "$SRC" ]]; then
  for cand in \
    "$ROOT/../../ereshkigal/vendor" \
    "${HOME}/Documents/_Projects/ereshkigal/vendor"
  do
    if [[ -d "$cand/vulkan-headers" ]]; then
      SRC="$cand"
      break
    fi
  done
fi
if [[ -z "$SRC" || ! -d "$SRC/vulkan-headers" ]]; then
  echo "no source vendor tree (set ERESHKIGAL_VENDOR_SRC)" >&2
  exit 1
fi
mkdir -p "$DEST"
for name in vulkan-headers spirv-prefix; do
  if [[ ! -d "$SRC/$name" ]]; then
    echo "skip missing $SRC/$name" >&2
    continue
  fi
  rm -rf "$DEST/$name"
  cp -a "$SRC/$name" "$DEST/$name"
  echo "vendored $name → $DEST/$name"
done
if [[ ! -f "$DEST/vulkan-headers/include/vulkan/vulkan.h" ]]; then
  echo "vulkan.h still missing under $DEST/vulkan-headers" >&2
  exit 1
fi

#!/usr/bin/env bash
# Local Vulkan llama.cpp build (no sudo). Needs libvulkan.so (vulkan-icd-loader).
# Headers: vendor/vulkan-headers + vendor/spirv-prefix (see scripts/setup_semif_venv.sh analogue).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HDR="$ROOT/vendor/vulkan-headers"
SPIRV="$ROOT/vendor/spirv-prefix"
if [[ ! -f "$HDR/include/vulkan/vulkan.h" ]]; then
  echo "missing $HDR/include/vulkan/vulkan.h (clone Khronos Vulkan-Headers)" >&2
  exit 1
fi
if [[ ! -f "$SPIRV/include/spirv/unified1/spirv.hpp" ]]; then
  echo "missing SPIRV headers under $SPIRV — cmake --install SPIRV-Headers" >&2
  exit 1
fi
export CARGO_TARGET_DIR="$ROOT/target-vulkan"
export CMAKE_PREFIX_PATH="${SPIRV}:${HDR}:${CMAKE_PREFIX_PATH:-}"
export VULKAN_SDK="$HDR"
export CPATH="${SPIRV}/include:${HDR}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${SPIRV}/include:${HDR}/include:${CPLUS_INCLUDE_PATH:-}"
export CMAKE_ARGS="-DVulkan_INCLUDE_DIR=${HDR}/include -DVulkan_LIBRARY=/usr/lib/libvulkan.so -DCMAKE_CXX_FLAGS=-I${SPIRV}/include -DCMAKE_C_FLAGS=-I${SPIRV}/include"
cd "$ROOT"
cargo build -p ereshkigal --release --features vulkan --bin semif-score --bin ereshkigal
echo "BIN=$CARGO_TARGET_DIR/release/semif-score"
echo "N_GPU_LAYERS=99 $CARGO_TARGET_DIR/release/semif-score --n-gpu-layers 99 ..."

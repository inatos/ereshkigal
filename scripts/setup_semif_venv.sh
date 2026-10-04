#!/usr/bin/env bash
# Isolated SemIf Python env for fair same-GGUF A/B (do not reuse training/.venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENDOR="${SEMIF_SRC:-$ROOT/vendor/SemIf-OpenJev}"
VENV="${SEMIF_VENV:-$ROOT/.venv-semif}"
export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -d "$VENDOR/.git" ]]; then
  mkdir -p "$(dirname "$VENDOR")"
  git clone --depth 1 https://github.com/TheoLeeCJ/SemIf-OpenJev.git "$VENDOR"
fi
if [[ -n "${SEMIF_PIN:-}" ]]; then
  git -C "$VENDOR" fetch --depth 1 origin "$SEMIF_PIN"
  git -C "$VENDOR" checkout "$SEMIF_PIN"
fi
ACTUAL=$(git -C "$VENDOR" rev-parse HEAD)
echo "SemIf commit $ACTUAL"

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12
fi
# pytest + llamacpp scoring without pinning SemIf's torch==2.10 (optional BF16 path).
uv pip install --python "$VENV/bin/python" \
  "pytest==8.4.2" \
  "llama-cpp-python==0.3.35" \
  "numpy>=2.0" \
  "huggingface-hub" \
  "tokenizers" \
  "transformers>=4.51" \
  "protobuf"
uv pip install --python "$VENV/bin/python" --no-deps -e "$VENDOR"

"$VENV/bin/python" -c "import semif_phase1, pytest, llama_cpp; print('semif', semif_phase1.__version__, 'pytest ok', 'llama_cpp', llama_cpp.__version__)"
"$VENV/bin/semif-score" --help >/dev/null
echo "SEMIF_SCORE=$VENV/bin/semif-score"
echo "Run unit tests: $VENV/bin/pytest $VENDOR/tests/test_core.py $VENDOR/tests/test_cli.py $VENDOR/tests/test_calibrate.py -q"

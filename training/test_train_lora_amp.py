#!/usr/bin/env python3
"""Tests for train_lora.py AMP dtype resolution and loss-head precision.

The risk this guards: `--amp bf16` on a Pascal 1080 Ti (sm_61) would advertise a
tensor rate that hardware cannot deliver, and a silent full-model downcast would
change BA/ECE numbers in a way that looks like a training regression rather than
a precision change. These tests pin the resolution rules without needing a GPU.

Run:
    tools/ereshkigal/training/.venv/bin/python -m pytest tools/ereshkigal/training/test_train_lora_amp.py
    # or, no pytest:
    tools/ereshkigal/training/.venv/bin/python tools/ereshkigal/training/test_train_lora_amp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_lora  # noqa: E402


class FakeTorch:
    """Minimal torch stand-in exposing only what _resolve_amp_dtype reads."""

    class bfloat16:  # noqa: N801 - mirror torch naming
        pass

    class float16:  # noqa: N801
        pass

    def __init__(self, bf16_supported: bool | Exception) -> None:
        self._bf16 = bf16_supported
        outer = self

        class _Cuda:
            def is_bf16_supported(_self):
                if isinstance(outer._bf16, Exception):
                    raise outer._bf16
                return outer._bf16

        self.cuda = _Cuda()


def test_auto_prefers_bf16_on_ampere():
    t = FakeTorch(bf16_supported=True)
    dtype, label = train_lora._resolve_amp_dtype(t, "cuda", "auto")
    assert dtype is FakeTorch.bfloat16, dtype
    assert label == "bf16 (auto)", label


def test_auto_falls_back_to_fp16_on_pascal():
    """sm_61 has no bf16 hardware; the label must say so, not pretend."""
    t = FakeTorch(bf16_supported=False)
    dtype, label = train_lora._resolve_amp_dtype(t, "cuda", "auto")
    assert dtype is FakeTorch.float16, dtype
    assert "bf16 unsupported" in label, label


def test_explicit_bf16_still_degrades_on_pascal():
    t = FakeTorch(bf16_supported=False)
    dtype, label = train_lora._resolve_amp_dtype(t, "cuda", "bf16")
    assert dtype is FakeTorch.float16, dtype
    assert "unsupported" in label, label


def test_explicit_fp16_is_respected():
    t = FakeTorch(bf16_supported=True)
    dtype, label = train_lora._resolve_amp_dtype(t, "cuda", "fp16")
    assert dtype is FakeTorch.float16, dtype
    assert label == "fp16", label


def test_off_disables_autocast():
    t = FakeTorch(bf16_supported=True)
    dtype, label = train_lora._resolve_amp_dtype(t, "cuda", "off")
    assert dtype is None, dtype
    assert "no autocast" in label, label


def test_cpu_disables_autocast():
    t = FakeTorch(bf16_supported=True)
    dtype, _ = train_lora._resolve_amp_dtype(t, "cpu", "auto")
    assert dtype is None, dtype


def test_bf16_probe_exception_does_not_crash():
    """An older torch without is_bf16_supported must not take the run down."""
    t = FakeTorch(bf16_supported=RuntimeError("no such method"))
    dtype, label = train_lora._resolve_amp_dtype(t, "cuda", "auto")
    assert dtype is FakeTorch.float16, dtype
    assert "unsupported" in label, label


def test_score_batch_defaults_to_no_amp():
    """The amp_dtype kwarg must default to None so existing callers are safe."""
    import inspect

    sig = inspect.signature(train_lora._score_batch)
    assert sig.parameters["amp_dtype"].default is None


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

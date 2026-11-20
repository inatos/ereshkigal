#!/usr/bin/env python3
"""Unit tests for train_lora.py (split, CAPS, option shuffle, bidirectional attn,
AMP auto, CosineHead .ehead Rust↔PyTorch parity, CLEF ablation) and serve_torch.py.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import random
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("train_lora", HERE / "train_lora.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

SERVE_TORCH_PATH = HERE.parent / "scripts" / "serve_torch.py"
serve_spec = importlib.util.spec_from_file_location("serve_torch", SERVE_TORCH_PATH)
serve_mod = importlib.util.module_from_spec(serve_spec)
assert serve_spec is not None and serve_spec.loader is not None
serve_spec.loader.exec_module(serve_mod)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def test_truncate_evidence_cap():
    words = " ".join(f"w{i}" for i in range(500))
    clipped = mod.truncate_evidence(words, 384)
    assert len(clipped.split()) == 384


def test_lineage_split_no_leak():
    rows = []
    for i in range(12):
        lid = f"defect-{(i // 3)}"
        rows.append(
            {
                "id": f"row-{i}",
                "lineage_id": lid,
                "family": "defect",
                "state": " ".join(f"t{j}" for j in range(10)),
                "split": "dev",
            }
        )
    train, hold = mod.freeze_split(rows, holdout_frac=0.34, seed=7, lineage_key="lineage_id")
    by_id = {r["id"]: r for r in rows}
    train_l = {by_id[i]["lineage_id"] for i in train}
    hold_l = {by_id[i]["lineage_id"] for i in hold}
    assert not (train_l & hold_l)
    assert train and hold


def test_prompt_truncates_state_before_options():
    row = {
        "question": "Promote?",
        "state": " ".join(f"token{i}" for i in range(500)),
        "options": [
            {"id": "promote", "description": "Promote"},
            {"id": "hold", "description": "Hold"},
        ],
        "label": 0,
    }
    prompt, offsets = mod._prompt_from_row_with_option_offsets(row)
    assert "A." in prompt or '"letter": "A"' in prompt or "Promote" in prompt
    # Options must still appear after truncated evidence.
    assert "Hold" in prompt or "hold" in prompt.lower()
    assert len(offsets) == 2
    assert offsets[0] < offsets[1]
    assert prompt[offsets[0]] == "}" and prompt[offsets[1]] == "}"


def test_caps_loss_brier_and_log():
    # Positive logged outcome (y=1.0): higher p_chosen gives strictly lower loss.
    p_good = [0.85, 0.10, 0.05]
    p_bad = [0.15, 0.70, 0.15]
    for rule in ("brier", "log", "both"):
        l_good = mod.caps_loss_py(p_good, 0, outcome=1.0, rule=rule)
        l_bad = mod.caps_loss_py(p_bad, 0, outcome=1.0, rule=rule)
        assert l_good < l_bad, f"rule={rule}: {l_good} vs {l_bad}"

    # Exact Brier and log formulas on chosen action.
    assert abs(mod.caps_loss_py([0.8, 0.2], 0, outcome=1.0, rule="brier") - 0.04) < 1e-9
    assert abs(mod.caps_loss_py([0.8, 0.2], 0, outcome=0.0, rule="brier") - 0.64) < 1e-9
    assert abs(mod.caps_loss_py([0.8, 0.2], 0, outcome=1.0, rule="log") - (-math.log(0.8))) < 1e-6
    assert abs(mod.caps_loss_py([0.8, 0.2], 0, outcome=0.0, rule="log") - (-math.log(0.2))) < 1e-6

    # Negative logged outcome (y=0.0): lower probability on failed action gives lower loss.
    assert mod.caps_loss_py([0.1, 0.9], 0, outcome=0.0, rule="both") < mod.caps_loss_py(
        [0.9, 0.1], 0, outcome=0.0, rule="both"
    )

    if _has_torch():
        import torch

        logits = torch.tensor([1.5, 0.2, -0.4], dtype=torch.float16, requires_grad=True)
        probs = torch.softmax(logits.float(), dim=0).detach().tolist()
        for rule in ("brier", "log", "both"):
            for y in (0.0, 1.0):
                lt = mod.caps_loss_torch(logits, 0, outcome=y, rule=rule)
                assert lt.dtype == torch.float32, f"expected fp32 head cast, got {lt.dtype}"
                lp = mod.caps_loss_py(probs, 0, outcome=y, rule=rule)
                assert abs(lt.item() - lp) < 1e-5, f"rule={rule} y={y}: {lt.item()} vs {lp}"
        lt = mod.caps_loss_torch(logits, 0, outcome=1.0, rule="both")
        lt.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_option_shuffle_remaps_label_and_preserves_ordinal():
    row = {
        "id": "r1",
        "question": "Pick backend?",
        "state": "evidence",
        "options": [
            {"id": "vulkan", "description": "Vulkan"},
            {"id": "cpu", "description": "CPU"},
            {"id": "metal", "description": "Metal"},
            {"id": "cuda", "description": "CUDA"},
        ],
        "label": 2,
    }
    t_probs = [0.05, 0.10, 0.80, 0.05]
    rng = random.Random(123)
    saw_moved = False
    for _ in range(16):
        srow, sprobs = mod.shuffle_row_options(row, rng, t_probs)
        assert srow["options"][srow["label"]]["id"] == "metal"
        assert sprobs is not None
        assert abs(sprobs[srow["label"]] - 0.80) < 1e-9
        if srow["label"] != 2:
            saw_moved = True
    assert saw_moved, "shuffle never moved label across 16 draws"

    # Ordinal / score rows MUST keep exact option order.
    ord_row = {
        "id": "ord-1",
        "type": "score",
        "question": "Rate quality 1..4",
        "state": "evidence",
        "options": [
            {"id": "1", "description": "1 - poor"},
            {"id": "2", "description": "2 - fair"},
            {"id": "3", "description": "3 - good"},
            {"id": "4", "description": "4 - great"},
        ],
        "label": 3,
    }
    assert mod.is_ordinal_row(ord_row)
    for _ in range(8):
        srow, sprobs = mod.shuffle_row_options(ord_row, rng, [0.05, 0.1, 0.25, 0.6])
        assert [o["id"] for o in srow["options"]] == ["1", "2", "3", "4"]
        assert srow["label"] == 3
        assert sprobs == [0.05, 0.1, 0.25, 0.6]


def test_bidirectional_attention_unmasks_full_attn_only():
    class DummyAttn:
        def __init__(self):
            self.is_causal = True

    class DummyCfg:
        is_causal = True

    class DummyModel:
        def __init__(self):
            self.config = DummyCfg()
            self.layers = {
                "model.layers.0.self_attn": DummyAttn(),
                "model.layers.1.linear_attn": DummyAttn(),
                "model.layers.2.self_attn": DummyAttn(),
                "model.layers.3.deltanet_attn": DummyAttn(),
            }

        def named_modules(self):
            yield ("", self)
            for k, v in self.layers.items():
                yield (k, v)

    m = DummyModel()
    info = mod.configure_bidirectional_attention(m, "bidirectional")
    assert info["mode"] == "bidirectional"
    assert info["unmasked_layers"] == 2
    assert m.layers["model.layers.0.self_attn"].is_causal is False
    assert m.layers["model.layers.2.self_attn"].is_causal is False
    # Recurrent/linear attention layers must stay causal.
    assert m.layers["model.layers.1.linear_attn"].is_causal is True
    assert m.layers["model.layers.3.deltanet_attn"].is_causal is True
    assert m.config.is_causal is False

    # Causal mode is a no-op.
    m2 = DummyModel()
    info_c = mod.configure_bidirectional_attention(m2, "causal")
    assert info_c["unmasked_layers"] == 0
    assert m2.layers["model.layers.0.self_attn"].is_causal is True

    # Mask check (pure Python): bidirectional unmasks j > i on non-pad, masks pad.
    bi = mod.build_attention_mask([[1, 1, 1, 0]], bidirectional=True)[0]
    assert bi[0][2] == 0.0, "bidirectional must allow token 0 -> token 2"
    assert bi[0][3] < -1e6, "bidirectional must still mask pad token 3"
    ca = mod.build_attention_mask([[1, 1, 1, 0]], bidirectional=False)[0]
    assert ca[0][2] < -1e6, "causal must mask future token 0 -> token 2"
    assert ca[2][0] == 0.0

    if _has_torch():
        import torch

        m2d = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        m4d = mod.build_attention_mask(m2d, bidirectional=True, dtype=torch.float32)
        assert tuple(m4d.shape) == (1, 1, 4, 4)
        assert m4d[0, 0, 0, 2].item() == 0.0
        assert m4d[0, 0, 0, 3].item() < -1e30


def test_amp_auto_resolution():
    assert mod.resolve_amp_dtype("auto", "cpu") == ("fp32", False)
    assert mod.resolve_amp_dtype("auto", "cuda", (6, 1)) == ("fp32", False)  # Pascal 1080 Ti
    assert mod.resolve_amp_dtype("auto", "cuda", (7, 5)) == ("fp16", True)  # Turing
    assert mod.resolve_amp_dtype("auto", "cuda", (8, 6)) == ("bf16", True)  # Ampere 3080 Ti
    assert mod.resolve_amp_dtype("none", "cuda", (8, 6)) == ("fp32", False)
    assert mod.resolve_amp_dtype("bf16", "cuda", (6, 1)) == ("bf16", True)


def test_cosine_ehead_rust_pytorch_parity():
    # Exact deterministic fixture matching ereshkigal-core/src/cosine_head.rs tests.
    hidden = 4
    head_dim = mod.HEAD_DIM
    w_q = [((i % 13) - 6.0) * 0.0625 for i in range(head_dim * hidden)]
    w_o = [((i % 17) - 8.0) * 0.03125 for i in range(head_dim * hidden)]
    theta = 0.5
    raw = mod.pack_ehead(w_q, w_o, hidden, theta, head_dim)
    assert raw[:8] == b"EHEAD01\x00"
    assert len(raw) == mod.EHEAD_HEADER_BYTES + 2 * head_dim * hidden * 4
    assert (
        hashlib.sha256(raw).hexdigest()
        == "003875af19c9748f51c6c2fe8d0c9e4b1ec8e269e142b62d9a09fb2baab0bec1"
    )

    with tempfile.TemporaryDirectory() as td:
        p = mod.export_ehead(Path(td) / "head.ehead", w_q, w_o, hidden, theta, head_dim)
        parsed = mod.unpack_ehead(p.read_bytes())
        assert parsed["hidden"] == hidden
        assert parsed["head_dim"] == head_dim
        assert abs(parsed["theta"] - theta) < 1e-6

    h_q = [0.25, -0.5, 0.75, 1.0]
    h_opts = [
        [0.5, 0.25, -0.25, 0.75],
        [-0.5, 0.75, 0.25, -0.25],
    ]
    rust_golden = [-0.00168752, 0.00767222]
    py_scores = [mod.cosine_score_pair_py(h_q, o, w_q, w_o, hidden, theta, head_dim) for o in h_opts]
    max_diff_py = max(abs(a - b) for a, b in zip(py_scores, rust_golden))
    assert max_diff_py < 1e-4, f"Python vs Rust cosine diff {max_diff_py} >= 1e-4"

    if _has_torch():
        import torch

        head_mod = mod.build_cosine_head(hidden, head_dim)
        head_mod = type(head_mod).from_ehead_bytes(raw)
        assert head_mod.to_ehead_bytes() == raw
        with torch.no_grad():
            # Pass fp16 inputs to verify fp32 head cast preserves <1e-4 parity.
            pt_scores = head_mod(
                torch.tensor(h_q, dtype=torch.float16),
                torch.tensor(h_opts, dtype=torch.float16),
            )
            assert pt_scores.dtype == torch.float32
            pt_list = pt_scores.tolist()
        max_diff_pt = max(abs(a - b) for a, b in zip(pt_list, rust_golden))
        assert max_diff_pt < 1e-4, f"PyTorch vs Rust cosine diff {max_diff_pt} >= 1e-4"


def test_clef_head_python_ablation():
    if not _has_torch():
        return
    import torch

    clef = mod.build_clef_head(16, proj_dim=32)
    h_q = torch.randn(16, dtype=torch.float16)
    h_opts = torch.randn(3, 16, dtype=torch.float16)
    out = clef(h_q, h_opts)
    assert tuple(out.shape) == (3,)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    loss = mod.caps_loss_torch(out, 1, outcome=1.0, rule="both")
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in clef.parameters())


def test_serve_torch_dry_run_and_non_stub_gate():
    # Without --dry-run-uniform and without --model, serve_torch MUST refuse (no silent stub).
    ap = serve_mod.build_arg_parser()
    args_no_model = ap.parse_args(["--attention", "bidirectional"])
    args_no_model.model = None
    try:
        serve_mod.build_backend_from_args(args_no_model)
        raise AssertionError("expected SystemExit when --model omitted without --dry-run-uniform")
    except SystemExit:
        pass

    # Explicit --dry-run-uniform serves uniform smoke responses cleanly.
    args_dry = ap.parse_args(["--dry-run-uniform", "--attention", "bidirectional"])
    backend = serve_mod.build_backend_from_args(args_dry)
    h = serve_mod.health_payload(backend)
    assert h["ok"] is True and h["stub"] is True and h["attention"] == "bidirectional"

    single = serve_mod.score_payload(
        backend,
        {"state": "ok", "question": "Q?", "options": ["yes", "no"]},
    )
    assert single["ok"] is True and single["stub"] is True and single["dry_run_uniform"] is True
    assert single["probabilities"] == [0.5, 0.5]

    batch = serve_mod.score_payload(
        backend,
        {
            "state": "ok",
            "questions": [
                {"question": "Q1?", "options": ["a", "b"]},
                {"question": "Q2?", "options": ["x", "y", "z"]},
            ],
        },
    )
    assert batch["ok"] is True and batch["count"] == 2
    assert len(batch["results"][1]["probabilities"]) == 3


if __name__ == "__main__":
    test_truncate_evidence_cap()
    test_lineage_split_no_leak()
    test_prompt_truncates_state_before_options()
    test_caps_loss_brier_and_log()
    test_option_shuffle_remaps_label_and_preserves_ordinal()
    test_bidirectional_attention_unmasks_full_attn_only()
    test_amp_auto_resolution()
    test_cosine_ehead_rust_pytorch_parity()
    test_clef_head_python_ablation()
    test_serve_torch_dry_run_and_non_stub_gate()
    print(f"ok (10 tests, torch={_has_torch()})")

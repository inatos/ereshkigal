#!/usr/bin/env python3
"""Unit smoke for lineage-safe freeze_split + evidence truncation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("train_lora", HERE / "train_lora.py")
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


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
    prompt = mod._prompt_from_row(row)
    assert "A." in prompt or '"letter": "A"' in prompt or "Promote" in prompt
    # Options must still appear after truncated evidence.
    assert "Hold" in prompt or "hold" in prompt.lower()


if __name__ == "__main__":
    test_truncate_evidence_cap()
    test_lineage_split_no_leak()
    test_prompt_truncates_state_before_options()
    print("ok")

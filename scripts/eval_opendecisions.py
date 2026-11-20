#!/usr/bin/env python3
"""OpenDecisions + Betwixt decree eval harness (Intelligence / Calibration / Capability)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def brier(probs: list[float], gold: int) -> float:
    return sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(probs))


def rps(probs: list[float], gold: int) -> float:
    n = len(probs)
    if n == 0:
        return 1.0
    cum_p = 0.0
    score = 0.0
    for k, p in enumerate(probs):
        cum_p += p
        cum_y = 1.0 if gold <= k else 0.0
        score += (cum_p - cum_y) ** 2
    return score / n


def nll(probs: list[float], gold: int) -> float:
    p = max(probs[gold] if gold < len(probs) else 0.0, 1e-12)
    return -math.log(p)


def ece(rows: list[tuple[list[float], int]], bins: int = 10) -> float:
    if not rows:
        return 0.0
    n = len(rows)
    bin_conf = [0.0] * bins
    bin_acc = [0.0] * bins
    bin_count = [0.0] * bins
    for probs, gold in rows:
        conf = max(probs) if probs else 0.0
        pred = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
        b = min(bins - 1, int(conf * bins))
        bin_conf[b] += conf
        bin_acc[b] += 1.0 if pred == gold else 0.0
        bin_count[b] += 1.0
    out = 0.0
    for b in range(bins):
        if bin_count[b] == 0:
            continue
        out += (bin_count[b] / n) * abs(bin_acc[b] / bin_count[b] - bin_conf[b] / bin_count[b])
    return out


def aurc(margins: list[float], correct: list[bool]) -> float:
    if len(margins) < 2:
        return 1.0 - (sum(correct) / max(1, len(correct)))
    idx = sorted(range(len(margins)), key=lambda i: margins[i], reverse=True)
    wrong = 0.0
    area = 0.0
    for k, i in enumerate(idx):
        if not correct[i]:
            wrong += 1.0
        area += wrong / (k + 1)
    return area / len(idx)


def evaluate(path: Path) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    labeled = []
    margins = []
    correct = []
    for r in rows:
        probs = r.get("probabilities") or r.get("probs") or []
        gold = r.get("gold_index")
        if gold is None or not probs:
            continue
        labeled.append((probs, int(gold)))
        pred = max(range(len(probs)), key=lambda i: probs[i])
        correct.append(pred == int(gold))
        s = sorted(probs, reverse=True)
        margins.append((s[0] - s[1]) if len(s) > 1 else s[0] if s else 0.0)
    if not labeled:
        return {"n": 0, "error": "no labeled rows"}
    ba_hits = sum(1 for c in correct if c) / len(correct)
    return {
        "n": len(labeled),
        "intelligence_ba": round(ba_hits, 4),
        "calibration_ece": round(ece(labeled), 4),
        "calibration_brier": round(sum(brier(p, g) for p, g in labeled) / len(labeled), 4),
        "calibration_rps": round(sum(rps(p, g) for p, g in labeled) / len(labeled), 4),
        "capability_nll": round(sum(nll(p, g) for p, g in labeled) / len(labeled), 4),
        "aurc": round(aurc(margins, correct), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="rows with probabilities + gold_index")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = evaluate(args.jsonl)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()

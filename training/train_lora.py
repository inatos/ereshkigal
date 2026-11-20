#!/usr/bin/env python3
"""LoRA distillation & CAPS/Cosine/CLEF training for Qwen3 / Gemma. Pascal (sm_61) friendly.

Strict train/holdout IDs (no leakage). Supports:
- --mode ce (gold option-restricted CE + optional teacher soft KL) and
  --mode caps (chosen-action proper scoring: Brier / log / both).
- --readout letter (GGUF letter logits), --readout cosine (.ehead EHEAD01
  exporter matching ereshkigal-core/src/cosine_head.rs), and --readout clef
  (Python-only contextualized label-embedding fusion ablation).
- --option-shuffle (per-epoch option shuffle preserving ordinal score order).
- --attention bidirectional (unmasks full self-attention layers during train;
  GGUF serve stays causal) with 4D additive non-causal padding mask.
- --amp auto (Ampere+ bf16, Turing/Volta fp16 + GradScaler, Pascal sm_61 & CPU
  fp32) with explicit fp32 readout head cast.

Evidence: BA/ECE on holdout only — never 1e-6, never authored144.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import struct
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
EHEAD_MAGIC = b"EHEAD01\x00"
EHEAD_HEADER_BYTES = 24
HEAD_DIM = 256


def _ba_ece_arrays(y_true: list[int], y_pred: list[int], confs: list[float]) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "BA": None, "ECE": None, "accuracy": None}
    classes = sorted(set(y_true))
    recalls = []
    for c in classes:
        idx = [i for i, t in enumerate(y_true) if t == c]
        hits = sum(1 for i in idx if y_pred[i] == c)
        recalls.append(hits / len(idx))
    ba = sum(recalls) / len(recalls) if recalls else 0.0
    bins = 5
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            i
            for i, c in enumerate(confs)
            if c >= lo and (c < hi or (b == bins - 1 and c <= hi))
        ]
        if not members:
            continue
        acc = sum(1 for i in members if y_pred[i] == y_true[i]) / len(members)
        conf = sum(confs[i] for i in members) / len(members)
        ece += (len(members) / n) * abs(acc - conf)
    acc = sum(1 for i in range(n) if y_true[i] == y_pred[i]) / n
    return {"n": n, "BA": round(ba, 4), "ECE": round(ece, 4), "accuracy": round(acc, 4)}


EVIDENCE_TOKEN_CAP = 384


def _approx_tokens(text: str) -> int:
    return len(str(text).split())


def truncate_evidence(text: str, max_tokens: int = EVIDENCE_TOKEN_CAP) -> str:
    """Cap evidence so SemIf options are never clipped at max_length=768."""
    words = str(text).split()
    if len(words) <= max_tokens:
        return str(text).strip()
    return " ".join(words[:max_tokens])


def freeze_split(
    gold_rows: list[dict],
    holdout_frac: float = 1.0 / 3.0,
    seed: int = 7,
    lineage_key: str | None = None,
) -> tuple[list[str], list[str]]:
    """Family-stratified holdout; optional lineage grouping prevents handoff/ledger leakage.

    When ``lineage_key`` is set (e.g. ``lineage_id``), every row that shares a lineage
    value is kept entirely in train *or* entirely in holdout.
    """
    rng = random.Random(seed)
    if lineage_key:
        by_lineage: dict[str, list[dict]] = defaultdict(list)
        for r in gold_rows:
            lid = str(r.get(lineage_key) or r.get("family") or r.get("id") or "unknown")
            by_lineage[lid].append(r)
        lineage_ids = list(by_lineage.keys())
        rng.shuffle(lineage_ids)
        # Cap holdout at ~holdout_frac — never "one per family" when that empties train
        # (unique lineage_id per row + many families used to put 100% in holdout).
        n_hold = max(1, int(round(len(lineage_ids) * holdout_frac)))
        n_hold = min(n_hold, max(1, len(lineage_ids) - 1))  # keep ≥1 train lineage when possible
        hold_lineages: set[str] = set()
        seen_fam: set[str] = set()
        for lid in lineage_ids:
            if len(hold_lineages) >= n_hold:
                break
            fam = str(by_lineage[lid][0].get("family") or "unknown")
            if fam not in seen_fam:
                hold_lineages.add(lid)
                seen_fam.add(fam)
        for lid in lineage_ids:
            if len(hold_lineages) >= n_hold:
                break
            hold_lineages.add(lid)
        hold_ids: set[str] = set()
        for lid in hold_lineages:
            for r in by_lineage[lid]:
                hold_ids.add(r["id"])
        all_ids = [r["id"] for r in gold_rows]
        hold = [i for i in all_ids if i in hold_ids]
        train = [i for i in all_ids if i not in hold_ids]
        return train, hold

    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in gold_rows:
        by_fam[str(r.get("family") or "unknown")].append(r)
    hold_ids = set()
    for fam, items in by_fam.items():
        items = list(items)
        rng.shuffle(items)
        # Always keep at least one holdout per family.
        hold_ids.add(items[0]["id"])
        n_extra = max(0, int(round(len(items) * holdout_frac)) - 1)
        for r in items[1 : 1 + n_extra]:
            hold_ids.add(r["id"])
    all_ids = [r["id"] for r in gold_rows]
    hold = [i for i in all_ids if i in hold_ids]
    train = [i for i in all_ids if i not in hold_ids]
    return train, hold


# Match ereshkigal-lang DIRECT_SYSTEM + dumps_pythonish + qwen3 thinking-off template.
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def _dumps_pythonish(value) -> str:
    """Match Rust dumps_pythonish / Python json.dumps default separators."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_dumps_pythonish(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{_dumps_pythonish(k)}: {_dumps_pythonish(v)}" for k, v in value.items()
            )
            + "}"
        )
    raise TypeError(f"unsupported prompt value type: {type(value)}")


_PROMPT_STYLE = "semif"


def _prompt_from_row_with_option_offsets(row: dict) -> tuple[str, list[int]]:
    """Render student prompt + exact character offset of each option end marker.

    For SemIf (`direct-options-v1`), records the exact character index of each
    option object's closing `}`, matching Rust `render_direct_prompt_with_option_offsets`.
    """
    state = truncate_evidence(row.get("state", ""), EVIDENCE_TOKEN_CAP)
    opts = list(row.get("options") or [])
    if _PROMPT_STYLE == "compact":
        lines = [
            f"Question: {row.get('question', '')}",
            f"State: {state}",
            "Options:",
        ]
        prefix = "\n".join(lines) + "\n"
        out = prefix
        offsets: list[int] = []
        for i, o in enumerate(opts):
            desc = o.get("description") or o.get("id") or ""
            out += f"{LETTERS[i]}. {desc}"
            offsets.append(len(out) - 1)
            out += "\n"
        out += "Answer:"
        return out, offsets

    system = DIRECT_SYSTEM.strip()
    user_prefix = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
    )
    parts = [
        f'{{"evidence": {_dumps_pythonish(state)}, '
        f'"criterion": {_dumps_pythonish(row.get("question", ""))}, '
        f'"options": ['
    ]
    user = parts[0]
    offsets = []
    for i, o in enumerate(opts):
        if i > 0:
            user += ", "
        desc = o.get("description") or o.get("id") or ""
        user += (
            f'{{"letter": {_dumps_pythonish(LETTERS[i])}, '
            f'"description": {_dumps_pythonish(desc)}}}'
        )
        offsets.append(len(user_prefix) + len(user) - 1)
    user += "]}"
    prompt = (
        f"{user_prefix}{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return prompt, offsets


def _prompt_from_row(row: dict) -> str:
    """Render student prompt; default SemIf matches Rust / GGUF scorer.

    Evidence is truncated to ``EVIDENCE_TOKEN_CAP`` *before* options so
    ``max_length=768`` never clips option letters on Pascal FP32.
    """
    prompt, _ = _prompt_from_row_with_option_offsets(row)
    return prompt


def _locate_option_tokens(
    offset_mapping: list[tuple[int, int]],
    opt_char_offsets: list[int],
    seq_len: int,
) -> list[int]:
    """Map strictly increasing option-end char offsets to covering token indices."""
    positions: list[int] = []
    search_from = 0
    for ch_off in opt_char_offsets:
        found = None
        for idx in range(search_from, min(seq_len, len(offset_mapping))):
            start, end = offset_mapping[idx]
            if start < end and start <= ch_off < end:
                found = idx
                break
        if found is None:
            # Fallback if prompt was truncated past max_length: clamp to last valid token.
            found = max(0, min(seq_len - 1, search_from))
        positions.append(found)
        search_from = min(seq_len - 1, found + 1)
    return positions


def is_ordinal_row(row: dict) -> bool:
    """Return True when option ordering carries ordinal/score meaning and must be preserved."""
    if row.get("ordinal") or row.get("preserve_option_order"):
        return True
    kind = str(row.get("type") or row.get("kind") or "").strip().lower()
    return kind in ("score", "ordinal", "scale", "rating")


def shuffle_row_options(
    row: dict,
    rng: random.Random,
    teacher_probs: list[float] | None = None,
) -> tuple[dict, list[float] | None]:
    """Per-epoch option shuffle that remaps label + teacher probs while preserving ordinal rows."""
    opts = list(row.get("options") or [])
    n = len(opts)
    if n < 2 or is_ordinal_row(row):
        return dict(row), (list(teacher_probs) if teacher_probs is not None else None)
    perm = list(range(n))
    rng.shuffle(perm)
    shuffled = [opts[i] for i in perm]
    out = dict(row)
    out["options"] = shuffled
    if "label" in row and row["label"] is not None:
        old_label = int(row["label"])
        if 0 <= old_label < n:
            out["label"] = perm.index(old_label)
    out["_perm"] = perm
    new_probs = None
    if teacher_probs is not None and len(teacher_probs) == n:
        new_probs = [teacher_probs[i] for i in perm]
    return out, new_probs


def _permute_row(row: dict, k: int, teacher_probs: list[float] | None) -> tuple[dict, list[float] | None]:
    """Cyclic rotate options by k; remap gold label and teacher probs."""
    opts = list(row.get("options") or [])
    n = len(opts)
    if n < 2 or is_ordinal_row(row):
        return row, teacher_probs
    k = k % n
    if k == 0:
        return row, teacher_probs
    rotated = opts[k:] + opts[:k]
    # Old index i lands at (i - k) mod n.
    old_label = int(row["label"])
    new_label = (old_label - k) % n
    out = dict(row)
    out["options"] = rotated
    out["label"] = new_label
    out["_perm_k"] = k
    new_probs = None
    if teacher_probs and len(teacher_probs) == n:
        new_probs = teacher_probs[k:] + teacher_probs[:k]
    return out, new_probs


def _letter_id_cache(tok, max_n: int = 16) -> list[int]:
    ids = []
    for i in range(max_n):
        tid = tok(LETTERS[i], add_special_tokens=False).input_ids
        if not tid:
            raise RuntimeError(f"tokenizer has no id for letter {LETTERS[i]}")
        ids.append(int(tid[0]))
    return ids


def _softmax_t(xs, dim: int = -1):
    import torch

    return torch.nn.functional.softmax(xs, dim=dim)


# ---------------------------------------------------------------------------
# Phase 5: Cosine readout (.ehead EHEAD01 format) + CLEF Python-only ablation
# ---------------------------------------------------------------------------


def pack_ehead(
    w_q: list[float],
    w_o: list[float],
    hidden: int,
    theta: float,
    head_dim: int = HEAD_DIM,
) -> bytes:
    """Serialize cosine readout weights matching Rust `CosineHead::load_ehead` (`EHEAD01\\0`)."""
    expected = head_dim * hidden
    if len(w_q) != expected or len(w_o) != expected:
        raise ValueError(
            f"ehead weight length mismatch: need {expected} each, got w_q={len(w_q)} w_o={len(w_o)}"
        )
    header = struct.pack("<8sIIfI", EHEAD_MAGIC, int(hidden), int(head_dim), float(theta), 0)
    body_q = struct.pack(f"<{expected}f", *(float(v) for v in w_q))
    body_o = struct.pack(f"<{expected}f", *(float(v) for v in w_o))
    return header + body_q + body_o


def unpack_ehead(data: bytes) -> dict:
    """Deserialize `EHEAD01\\0` binary payload into `{hidden, head_dim, theta, w_q, w_o}`."""
    if len(data) < EHEAD_HEADER_BYTES:
        raise ValueError(f"ehead too short: {len(data)} < {EHEAD_HEADER_BYTES}")
    magic, hidden, head_dim, theta, _reserved = struct.unpack_from("<8sIIfI", data, 0)
    if magic != EHEAD_MAGIC:
        raise ValueError(f"ehead invalid magic: {magic!r} != {EHEAD_MAGIC!r}")
    count = head_dim * hidden
    need = EHEAD_HEADER_BYTES + 2 * count * 4
    if len(data) < need:
        raise ValueError(f"ehead truncated: got {len(data)} bytes, need {need}")
    w_q = list(struct.unpack_from(f"<{count}f", data, EHEAD_HEADER_BYTES))
    w_o = list(struct.unpack_from(f"<{count}f", data, EHEAD_HEADER_BYTES + count * 4))
    return {
        "hidden": int(hidden),
        "head_dim": int(head_dim),
        "theta": float(theta),
        "w_q": w_q,
        "w_o": w_o,
    }


def export_ehead(
    path: Path,
    w_q: list[float],
    w_o: list[float],
    hidden: int,
    theta: float,
    head_dim: int = HEAD_DIM,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pack_ehead(w_q, w_o, hidden, theta, head_dim=head_dim))
    return path


def _rms_norm_py(x: list[float]) -> list[float]:
    mean_sq = sum(v * v for v in x) / max(1, len(x))
    inv = 1.0 / math.sqrt(mean_sq + 1e-6)
    return [v * inv for v in x]


def cosine_score_pair_py(
    h_query: list[float],
    h_option: list[float],
    w_q: list[float],
    w_o: list[float],
    hidden: int,
    theta: float,
    head_dim: int = HEAD_DIM,
) -> float:
    """Pure-Python reference matching Rust `CosineHead::score_pair`."""
    if len(h_query) != hidden or len(h_option) != hidden:
        raise ValueError("hidden size mismatch")
    hq = _rms_norm_py(h_query)
    ho = _rms_norm_py(h_option)
    q_raw = [
        sum(w_q[i * hidden + j] * hq[j] for j in range(hidden)) for i in range(head_dim)
    ]
    o_raw = [
        sum(w_o[i * hidden + j] * ho[j] for j in range(hidden)) for i in range(head_dim)
    ]
    q = _rms_norm_py(q_raw)
    o = _rms_norm_py(o_raw)
    dot = sum(a * b for a, b in zip(q, o))
    scale = min(math.exp(float(theta)), 100.0) / float(head_dim)
    return scale * dot


def build_cosine_head(hidden: int, head_dim: int = HEAD_DIM):
    """Build PyTorch `CosineHeadModule` (always computes readout in fp32)."""
    import torch

    class CosineHeadModule(torch.nn.Module):
        def __init__(self, h: int, d: int):
            super().__init__()
            self.hidden = h
            self.head_dim = d
            self.w_q = torch.nn.Linear(h, d, bias=False)
            self.w_o = torch.nn.Linear(h, d, bias=False)
            self.theta = torch.nn.Parameter(torch.zeros(()))
            torch.nn.init.uniform_(self.w_q.weight, -0.01, 0.01)
            torch.nn.init.uniform_(self.w_o.weight, -0.01, 0.01)

        @staticmethod
        def rms_norm(x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        def forward(self, h_query: torch.Tensor, h_options: torch.Tensor) -> torch.Tensor:
            hq = self.rms_norm(h_query.float())
            ho = self.rms_norm(h_options.float())
            q = self.rms_norm(torch.nn.functional.linear(hq, self.w_q.weight.float()))
            o = self.rms_norm(torch.nn.functional.linear(ho, self.w_o.weight.float()))
            if o.dim() == 1:
                dot = (q * o).sum(dim=-1)
            else:
                dot = (o * q.unsqueeze(0)).sum(dim=-1)
            scale = torch.clamp(torch.exp(self.theta.float()), max=100.0) / float(self.head_dim)
            return scale * dot

        def to_ehead_bytes(self) -> bytes:
            wq = self.w_q.weight.detach().cpu().float().contiguous().view(-1).tolist()
            wo = self.w_o.weight.detach().cpu().float().contiguous().view(-1).tolist()
            return pack_ehead(wq, wo, self.hidden, float(self.theta.detach().cpu().item()), self.head_dim)

        def save_ehead(self, path: Path) -> Path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(self.to_ehead_bytes())
            return p

        @classmethod
        def from_ehead_bytes(cls, data: bytes):
            parsed = unpack_ehead(data)
            mod = cls(parsed["hidden"], parsed["head_dim"])
            with torch.no_grad():
                mod.w_q.weight.copy_(
                    torch.tensor(parsed["w_q"], dtype=torch.float32).view(
                        parsed["head_dim"], parsed["hidden"]
                    )
                )
                mod.w_o.weight.copy_(
                    torch.tensor(parsed["w_o"], dtype=torch.float32).view(
                        parsed["head_dim"], parsed["hidden"]
                    )
                )
                mod.theta.fill_(float(parsed["theta"]))
            return mod

    return CosineHeadModule(hidden, head_dim)


def load_cosine_head_ehead(path: Path):
    """Load a `.ehead` (`EHEAD01\\0`) file into a PyTorch `CosineHeadModule`."""
    data = Path(path).read_bytes()
    parsed = unpack_ehead(data)
    mod = build_cosine_head(parsed["hidden"], parsed["head_dim"])
    return type(mod).from_ehead_bytes(data)


def build_clef_head(hidden: int, proj_dim: int = HEAD_DIM):
    """Python-only CLEF ablation head (contextualized label-embedding fusion, fp32)."""
    import torch

    class ClefHeadModule(torch.nn.Module):
        def __init__(self, h: int, d: int):
            super().__init__()
            self.hidden = h
            self.proj_dim = d
            self.q_proj = torch.nn.Linear(h, d, bias=False)
            self.o_proj = torch.nn.Linear(h, d, bias=False)
            self.fuse = torch.nn.Sequential(
                torch.nn.Linear(3 * d, d),
                torch.nn.SiLU(),
                torch.nn.Linear(d, 1),
            )

        @staticmethod
        def rms_norm(x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        def forward(self, h_query: torch.Tensor, h_options: torch.Tensor) -> torch.Tensor:
            hq = self.rms_norm(h_query.float())
            ho = self.rms_norm(h_options.float())
            qp = self.q_proj(hq)  # [D]
            op = self.o_proj(ho)  # [N, D]
            qp_exp = qp.unsqueeze(0).expand_as(op)
            fused = torch.cat([qp_exp, op, qp_exp * op], dim=-1)  # [N, 3D]
            return self.fuse(fused).squeeze(-1)  # [N]

    return ClefHeadModule(hidden, proj_dim)


# ---------------------------------------------------------------------------
# Phase 6: CAPS loss, bidirectional unmasking, AMP auto resolution
# ---------------------------------------------------------------------------


def caps_loss_py(
    probs: list[float],
    chosen_idx: int,
    outcome: float = 1.0,
    rule: str = "both",
    eps: float = 1e-7,
) -> float:
    """CAPS chosen-action proper scoring rule (Brier / log / both)."""
    p_a = float(probs[chosen_idx])
    y = float(outcome)
    p_c = min(max(p_a, eps), 1.0 - eps)
    brier = (p_a - y) ** 2
    log_s = -(y * math.log(p_c) + (1.0 - y) * math.log(1.0 - p_c))
    if rule == "brier":
        return brier
    if rule == "log":
        return log_s
    if rule == "both":
        return 0.5 * brier + 0.5 * log_s
    raise ValueError(f"unknown caps rule {rule!r}")


def caps_loss_torch(
    option_logits,
    chosen_idx: int,
    outcome: float = 1.0,
    rule: str = "both",
    eps: float = 1e-7,
):
    """Differentiable CAPS proper scoring loss on chosen action (explicit fp32 head cast)."""
    import torch

    logits_f32 = option_logits.float()
    probs = torch.softmax(logits_f32, dim=-1)
    p_a = probs[int(chosen_idx)]
    y = torch.tensor(float(outcome), device=logits_f32.device, dtype=torch.float32)
    brier = (p_a - y).pow(2)
    p_c = p_a.clamp(min=eps, max=1.0 - eps)
    log_s = -(y * torch.log(p_c) + (1.0 - y) * torch.log(1.0 - p_c))
    if rule == "brier":
        return brier
    if rule == "log":
        return log_s
    if rule == "both":
        return 0.5 * brier + 0.5 * log_s
    raise ValueError(f"unknown caps rule {rule!r}")


def configure_bidirectional_attention(model, attention_mode: str) -> dict:
    """Unmask full self-attention layers (`is_causal = False`) when `bidirectional`.

    Skips recurrent/linear-attention layers (e.g. Qwen3.5 `linear_attn` / DeltaNet)
    so only full self-attention layers are unmasked. GGUF serve stays causal.
    """
    if attention_mode != "bidirectional":
        return {"mode": "causal", "unmasked_layers": 0, "patched_modules": []}
    patched: list[str] = []
    for name, mod in model.named_modules():
        low = name.lower()
        if "linear_attn" in low or "deltanet" in low:
            continue
        if hasattr(mod, "is_causal"):
            try:
                setattr(mod, "is_causal", False)
                patched.append(name or type(mod).__name__)
            except Exception:
                pass
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "is_causal"):
        try:
            cfg.is_causal = False
        except Exception:
            pass
    return {
        "mode": "bidirectional",
        "unmasked_layers": len(patched),
        "patched_modules": patched,
    }


def build_attention_mask(attention_mask_2d, bidirectional: bool, dtype=None):
    """Build explicit additive attention mask (4D for torch, 3D [B][T][T] for Python lists).

    - `bidirectional=True`: valid tokens attend to ALL valid tokens (including `j > i`);
      only pad tokens (`mask == 0`) are masked out.
    - `bidirectional=False`: standard causal lower-triangular + padding mask.
    """
    if isinstance(attention_mask_2d, (list, tuple)):
        neg = -1e9
        out = []
        for row in attention_mask_2d:
            t = len(row)
            mat = []
            for i in range(t):
                r = []
                for j in range(t):
                    valid_k = bool(row[j])
                    allowed = valid_k and (bidirectional or j <= i)
                    r.append(0.0 if allowed else neg)
                mat.append(r)
            out.append(mat)
        return out

    import torch

    if not bidirectional:
        return attention_mask_2d
    bsz, seq_len = attention_mask_2d.shape
    mask_dtype = dtype or torch.float32
    min_val = torch.finfo(mask_dtype).min
    # [B, 1, 1, T] broadcast to [B, 1, T, T]: 0.0 for valid keys, min_val for padded keys.
    key_valid = attention_mask_2d.to(dtype=torch.bool)[:, None, None, :]
    mask_4d = torch.zeros((bsz, 1, seq_len, seq_len), device=attention_mask_2d.device, dtype=mask_dtype)
    mask_4d = mask_4d.masked_fill(~key_valid, min_val)
    return mask_4d


def resolve_amp_dtype(
    amp_mode: str,
    device: str,
    cuda_capability: tuple[int, int] | None = None,
) -> tuple[str, bool]:
    """Resolve `(dtype_name, autocast_enabled)` with Pascal sm_61 safety.

    Returns one of `("fp32", False)`, `("fp16", True)`, `("bf16", True)`.
    Readout heads are always cast to `fp32` regardless of backbone autocast dtype.
    """
    mode = (amp_mode or "auto").strip().lower()
    if mode == "none" or device != "cuda":
        return ("fp32", False)
    if mode == "bf16":
        return ("bf16", True)
    if mode == "fp16":
        return ("fp16", True)
    if mode == "auto":
        if cuda_capability is None:
            try:
                import torch

                if torch.cuda.is_available():
                    cuda_capability = torch.cuda.get_device_capability(0)
            except Exception:
                cuda_capability = None
        if cuda_capability is None:
            return ("fp32", False)
        major, _minor = cuda_capability
        if major >= 8:
            return ("bf16", True)
        if major >= 7:
            return ("fp16", True)
        # Pascal (sm_61, e.g. GTX 1080 Ti) has weak fp16 / no bf16 → stay fp32.
        return ("fp32", False)
    raise ValueError(f"unsupported amp mode {amp_mode!r}")


def _build_examples(
    train_rows: list[dict],
    teacher_by_id: dict,
    permute: bool,
    upsample_families: set[str] | None = None,
    upsample_factor: int = 2,
    option_shuffle: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    ex = []
    up = upsample_families or set()
    shuf_rng = rng or random.Random(7)
    for row in train_rows:
        tid = row["id"]
        soft = None
        t = teacher_by_id.get(tid)
        if t:
            soft = t.get("probabilities") or t.get("teacher_probs")
            if soft is not None:
                soft = [float(x) for x in soft]
        n = len(row.get("options") or [])
        fam = str(row.get("family") or "")
        reps = upsample_factor if fam in up else 1
        outcome = float(row.get("outcome", row.get("reward", 1.0)))
        weight = float(row.get("rlsf_weight", row.get("advantage", 1.0)))
        for _ in range(reps):
            if option_shuffle:
                prow, psoft = shuffle_row_options(row, shuf_rng, soft)
                prompt, opt_offsets = _prompt_from_row_with_option_offsets(prow)
                ex.append(
                    {
                        "id": tid,
                        "prompt": prompt,
                        "opt_offsets": opt_offsets,
                        "label": int(prow["label"]),
                        "n_opts": len(prow.get("options") or []),
                        "teacher_probs": psoft,
                        "outcome": outcome,
                        "weight": weight,
                    }
                )
            else:
                cycles = list(range(n)) if permute and n >= 2 and not is_ordinal_row(row) else [0]
                for k in cycles:
                    prow, psoft = _permute_row(row, k, soft)
                    prompt, opt_offsets = _prompt_from_row_with_option_offsets(prow)
                    ex.append(
                        {
                            "id": tid,
                            "prompt": prompt,
                            "opt_offsets": opt_offsets,
                            "label": int(prow["label"]),
                            "n_opts": len(prow.get("options") or []),
                            "teacher_probs": psoft,
                            "outcome": outcome,
                            "weight": weight,
                        }
                    )
    return ex


def _forward_option_logits(
    model,
    tok,
    device: str,
    letter_ids: list[int],
    prompts: list[str],
    opt_offsets_batch: list[list[int]],
    n_opts_list: list[int],
    max_length: int,
    readout: str,
    head_module,
    bidirectional: bool,
    amp_kind: str,
    use_amp: bool,
):
    """Run one batched forward pass; returns list of per-example fp32 `[n_opts]` tensors."""
    import torch

    need_hidden = readout in ("cosine", "clef")
    enc = tok(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
        return_offsets_mapping=need_hidden,
    )
    offset_mappings = enc.pop("offset_mapping", None)
    orig_attn = enc["attention_mask"].to(device)
    model_inputs = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": orig_attn,
    }
    if bidirectional:
        mask_dtype = (
            torch.bfloat16
            if (use_amp and amp_kind == "bf16")
            else (torch.float16 if (use_amp and amp_kind == "fp16") else torch.float32)
        )
        model_inputs["attention_mask"] = build_attention_mask(
            orig_attn, bidirectional=True, dtype=mask_dtype
        )

    if use_amp:
        amp_dtype = torch.bfloat16 if amp_kind == "bf16" else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        out = model(**model_inputs, output_hidden_states=need_hidden)

    last = orig_attn.sum(dim=1) - 1  # [B]
    bsz = orig_attn.size(0)
    per_row_logits = []
    if need_hidden:
        # Always cast hidden states to fp32 before readout head (--amp auto fp32 head cast).
        hidden = out.hidden_states[-1].float()  # [B, T, H]
        for i in range(bsz):
            seq_len = int(orig_attn[i].sum().item())
            om = offset_mappings[i].tolist() if offset_mappings is not None else []
            opt_pos = _locate_option_tokens(om, opt_offsets_batch[i], seq_len)
            h_q = hidden[i, last[i]]  # [H]
            h_opts = hidden[i, opt_pos]  # [n, H]
            per_row_logits.append(head_module(h_q, h_opts).float())
    else:
        # Explicit fp32 head cast on gathered letter logits.
        logits = out.logits
        gathered = logits[torch.arange(bsz, device=device), last].float()  # [B, V]
        for i in range(bsz):
            n = n_opts_list[i]
            per_row_logits.append(gathered[i, letter_ids[:n]])
    return per_row_logits


def _score_batch(
    model,
    tok,
    device,
    letter_ids: list[int],
    rows: list[dict],
    batch_size: int = 4,
    max_length: int = 768,
    readout: str = "letter",
    head_module=None,
    bidirectional: bool = False,
    amp_kind: str = "fp32",
    use_amp: bool = False,
):
    """Batched scoring for letter, cosine, or clef readouts. Returns pred/conf/label arrays."""
    import torch

    model.eval()
    if head_module is not None:
        head_module.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confs: list[float] = []
    rendered = [_prompt_from_row_with_option_offsets(r) for r in rows]
    prompts = [p for p, _ in rendered]
    opt_offsets = [o for _, o in rendered]
    labels = [int(r["label"]) for r in rows]
    n_opts_list = [len(r.get("options") or []) for r in rows]

    for start in range(0, len(prompts), batch_size):
        chunk_p = prompts[start : start + batch_size]
        chunk_o = opt_offsets[start : start + batch_size]
        chunk_n = n_opts_list[start : start + batch_size]
        chunk_y = labels[start : start + batch_size]
        with torch.no_grad():
            row_logits = _forward_option_logits(
                model,
                tok,
                device,
                letter_ids,
                chunk_p,
                chunk_o,
                chunk_n,
                max_length,
                readout,
                head_module,
                bidirectional,
                amp_kind,
                use_amp,
            )
        for i, scores in enumerate(row_logits):
            probs = _softmax_t(scores.float(), dim=0)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
            y_true.append(chunk_y[i])
            y_pred.append(pred)
            confs.append(conf)
    return y_true, y_pred, confs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="Qwen/Qwen3-0.6B")
    p.add_argument("--teacher-jsonl", type=Path, required=False)
    p.add_argument("--gold", type=Path, help="dev_gold.jsonl for train/holdout")
    p.add_argument(
        "--extra-gold",
        type=Path,
        action="append",
        default=[],
        help="Additional gold JSONL (e.g. ereshkigal_handoff_gold.jsonl); may repeat",
    )
    p.add_argument(
        "--lineage-key",
        default="lineage_id",
        help="Group holdout by this field to stop handoff/ledger leakage (empty disables)",
    )
    p.add_argument(
        "--rlsf-weights",
        action="store_true",
        help="Weight loss by row rlsf_weight / advantage when present",
    )
    p.add_argument("--out", type=Path, default=Path("packs/qwen3-0.6b-lora"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="comma-separated LoRA target modules",
    )
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--ce-weight", type=float, default=0.6)
    p.add_argument("--kl-weight", type=float, default=0.4)
    p.add_argument("--kl-temp", type=float, default=2.0)
    p.add_argument("--permute", action="store_true", default=True)
    p.add_argument("--no-permute", action="store_false", dest="permute")
    p.add_argument(
        "--upsample-families",
        default="policy,contrast",
        help="comma-separated families to upsample (empty to disable)",
    )
    p.add_argument("--upsample-factor", type=int, default=2)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument(
        "--split-json",
        type=Path,
        help="load frozen holdout IDs if present; always rewrite with current train/holdout",
    )
    p.add_argument(
        "--prompt-style",
        choices=("semif", "compact"),
        default="semif",
        help="semif = Rust direct-options-v1; compact = Question/State/Options/Answer",
    )
    p.add_argument(
        "--init-adapter",
        type=Path,
        help="Continue training from an existing PEFT pack (trainable)",
    )
    p.add_argument(
        "--mode",
        default="ce",
        choices=("ce", "caps"),
        help="ce = option CE/KL; caps = chosen-action proper scoring (Brier/log)",
    )
    p.add_argument(
        "--caps-rule",
        default="both",
        choices=("both", "brier", "log"),
        help="proper scoring rule for --mode caps (default: 0.5*brier + 0.5*log)",
    )
    p.add_argument(
        "--readout",
        default="letter",
        choices=("letter", "cosine", "clef"),
        help="letter logits (GGUF); cosine .ehead; clef=python ablation only",
    )
    p.add_argument(
        "--attention",
        default="causal",
        choices=("causal", "bidirectional"),
        help="bidirectional unmasks full-attn layers in train; GGUF serve stays causal",
    )
    p.add_argument(
        "--amp",
        default="auto",
        choices=("auto", "none", "bf16", "fp16"),
        help="mixed precision autocast (auto: bf16 Ampere+, fp16 sm_70+, fp32 Pascal/CPU); readout head always fp32",
    )
    p.add_argument(
        "--option-shuffle",
        action="store_true",
        help="per-epoch option shuffle (preserve ordinal score order)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 11, 13],
        help="Jeff-1.1 soup seeds (use soup_lora.py after)",
    )
    p.add_argument(
        "--qwen-linear-attn-targets",
        action="store_true",
        help="also target in_proj_qkv,in_proj_z,out_proj for Qwen linear-attn",
    )
    args = p.parse_args()
    if args.qwen_linear_attn_targets:
        extra = ["in_proj_qkv", "in_proj_z", "out_proj"]
        cur = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
        for e in extra:
            if e not in cur:
                cur.append(e)
        args.lora_targets = ",".join(cur)
    args.out.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "student": args.student,
        "teacher_jsonl": str(args.teacher_jsonl) if args.teacher_jsonl else None,
        "gold": str(args.gold) if args.gold else None,
        "steps": args.steps,
        "lr": args.lr,
        "mode": args.mode,
        "caps_rule": args.caps_rule if args.mode == "caps" else None,
        "readout": args.readout,
        "attention": args.attention,
        "amp": args.amp,
        "head_dtype": "float32",
        "option_shuffle": bool(args.option_shuffle),
        "ce_weight": args.ce_weight,
        "kl_weight": args.kl_weight,
        "kl_temp": args.kl_temp,
        "permute": args.permute,
        "note": (
            "strict split; SemIf direct-options-v1 prompts (Rust-parity); "
            "CE or CAPS proper scoring; BA/ECE holdout only; never 1e-6/authored144"
        ),
        "prompt_version": (
            "direct-options-v1" if args.prompt_style == "semif" else "compact-student-v1"
        ),
        "prompt_style": args.prompt_style,
        "status": "scaffolded",
        "gate_BA": 0.75,
    }
    if args.readout == "clef":
        meta["python_only_ablation"] = True
    global _PROMPT_STYLE
    _PROMPT_STYLE = args.prompt_style

    gold_rows: list[dict] = []
    gold_paths: list[Path] = []
    if args.gold and args.gold.is_file():
        gold_paths.append(args.gold)
    for extra in args.extra_gold or []:
        if extra and Path(extra).is_file():
            gold_paths.append(Path(extra))
    for path in gold_paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                continue
            row.setdefault("split", "dev")
            if "state" in row:
                row["state"] = truncate_evidence(row["state"], EVIDENCE_TOKEN_CAP)
            gold_rows.append(row)
    if gold_rows:
        assert all(r.get("split") == "dev" for r in gold_rows), "gold must be split=dev"
    # Deduplicate by id (first wins) so extra-gold does not clobber frozen gold.
    dedup: dict[str, dict] = {}
    for r in gold_rows:
        dedup.setdefault(r["id"], r)
    gold_rows = list(dedup.values())
    gold_by_id = {r["id"]: r for r in gold_rows}
    meta["extra_gold"] = [str(p) for p in gold_paths]
    meta["lineage_key"] = args.lineage_key or None
    meta["rlsf_weights"] = bool(args.rlsf_weights)
    meta["evidence_token_cap"] = EVIDENCE_TOKEN_CAP

    teacher_rows: list[dict] = []
    if args.teacher_jsonl and args.teacher_jsonl.is_file():
        teacher_rows = [
            json.loads(l)
            for l in args.teacher_jsonl.read_text().splitlines()
            if l.strip()
        ]
        meta["n_teacher"] = len(teacher_rows)
    teacher_by_id = {r["id"]: r for r in teacher_rows if r.get("id")}

    train_ids, hold_ids = ([], [])
    if gold_rows:
        if args.split_json and args.split_json.is_file():
            frozen = json.loads(args.split_json.read_text())
            hold_ids = list(frozen.get("holdout_ids") or [])
            hold_set = set(hold_ids)
            all_ids = [r["id"] for r in gold_rows]
            # Keep frozen holdout; any new IDs join train (never leak holdout).
            train_ids = [i for i in all_ids if i not in hold_set]
            missing = [i for i in hold_ids if i not in set(all_ids)]
            if missing:
                raise SystemExit(f"frozen holdout IDs missing from gold: {missing}")
            meta["split_source"] = str(args.split_json)
        else:
            lineage = args.lineage_key.strip() if args.lineage_key else None
            train_ids, hold_ids = freeze_split(
                gold_rows, seed=args.split_seed, lineage_key=lineage or None
            )
            meta["split_source"] = "freeze_split"
            # Lineage leak guard: no lineage_id in both train and holdout.
            if lineage:
                train_l = {
                    str(gold_by_id[i].get(lineage) or i) for i in train_ids if i in gold_by_id
                }
                hold_l = {
                    str(gold_by_id[i].get(lineage) or i) for i in hold_ids if i in gold_by_id
                }
                assert not (train_l & hold_l), "train/holdout lineage overlap"
    meta["train_ids"] = train_ids
    meta["holdout_ids"] = hold_ids
    meta["n_train"] = len(train_ids)
    meta["n_holdout"] = len(hold_ids)
    if args.split_json:
        args.split_json.parent.mkdir(parents=True, exist_ok=True)
        args.split_json.write_text(
            json.dumps(
                {
                    "train_ids": train_ids,
                    "holdout_ids": hold_ids,
                    "seed": args.split_seed,
                    "n": len(gold_rows),
                },
                indent=2,
            )
            + "\n"
        )

    hold_rows = [gold_by_id[i] for i in hold_ids if i in gold_by_id]
    train_rows = [gold_by_id[i] for i in train_ids if i in gold_by_id]
    # Leak guard: never train on holdout IDs.
    assert not (set(train_ids) & set(hold_ids)), "train/holdout ID overlap"

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["status"] = "blocked"
        meta["blocked_reason"] = (
            f"training stack import failed ({type(e).__name__}: {e}); "
            "pin peft==0.13.2 + transformers==4.51.3 for torch 2.4.1+cu118"
        )
        (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("training extras missing:", e)
        return

    meta["torch"] = torch.__version__
    meta["cuda"] = torch.cuda.is_available()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    cuda_cap = torch.cuda.get_device_capability(0) if device == "cuda" else None
    amp_kind, use_amp = resolve_amp_dtype(args.amp, device, cuda_cap)
    meta["device"] = device
    meta["amp_resolved"] = amp_kind
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda_capability"] = list(cuda_cap) if cuda_cap else None

    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=targets)
    meta["lora"] = {"r": args.lora_r, "alpha": args.lora_alpha, "targets": targets}

    if args.steps <= 0:
        meta["status"] = "smoke"
        (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("smoke only", meta["torch"], meta["cuda"], "amp=", amp_kind)
        return

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = _letter_id_cache(tok, 16)

    base = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.float32, trust_remote_code=True
    )
    base.to(device)

    bidirectional = args.attention == "bidirectional"
    attn_patch = configure_bidirectional_attention(base, args.attention)
    meta["attention_patch"] = attn_patch

    head_module = None
    if args.readout == "cosine":
        hidden_size = int(getattr(base.config, "hidden_size"))
        head_module = build_cosine_head(hidden_size, HEAD_DIM).to(device=device, dtype=torch.float32)
    elif args.readout == "clef":
        hidden_size = int(getattr(base.config, "hidden_size"))
        head_module = build_clef_head(hidden_size, HEAD_DIM).to(device=device, dtype=torch.float32)

    # Unadapted student baseline on holdout (letter readout baseline).
    if hold_rows:
        y_t, y_p, confs = _score_batch(
            base,
            tok,
            device,
            letter_ids,
            hold_rows,
            max_length=args.max_length,
            readout=args.readout,
            head_module=head_module,
            bidirectional=bidirectional,
            amp_kind=amp_kind,
            use_amp=use_amp,
        )
        meta["unadapted_holdout"] = _ba_ece_arrays(y_t, y_p, confs)

    if args.init_adapter:
        from peft import PeftModel

        init_path = args.init_adapter
        if not (init_path / "adapter_config.json").is_file():
            meta["status"] = "blocked"
            meta["blocked_reason"] = f"init-adapter missing adapter_config.json: {init_path}"
            (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
            return
        model = PeftModel.from_pretrained(base, str(init_path), is_trainable=True)
        meta["init_adapter"] = str(init_path)
        print(f"continue from adapter {init_path}")
    else:
        model = get_peft_model(base, cfg)
    if bidirectional:
        # Ensure wrapped PEFT model layers retain is_causal = False.
        configure_bidirectional_attention(model, "bidirectional")

    model.train()
    trainable = [x for x in model.parameters() if x.requires_grad]
    if head_module is not None:
        head_module.train()
        trainable.extend(x for x in head_module.parameters() if x.requires_grad)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_kind == "fp16"))

    up_fams = {f.strip() for f in args.upsample_families.split(",") if f.strip()}
    rng = random.Random(args.split_seed)
    examples = _build_examples(
        train_rows,
        teacher_by_id,
        args.permute,
        upsample_families=up_fams,
        upsample_factor=max(1, args.upsample_factor),
        option_shuffle=args.option_shuffle,
        rng=rng,
    )
    meta["n_examples_aug"] = len(examples)
    meta["upsample_families"] = sorted(up_fams)
    meta["upsample_factor"] = args.upsample_factor
    meta["max_length"] = args.max_length
    if not examples:
        meta["status"] = "blocked"
        meta["blocked_reason"] = "no train examples after split"
        (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
        return

    mb = max(1, args.microbatch)
    ce_w, kl_w, T = args.ce_weight, args.kl_weight, args.kl_temp
    hold_set = set(hold_ids)
    steps_done = 0
    epoch = 0

    try:
        while steps_done < args.steps:
            if args.option_shuffle and epoch > 0:
                examples = _build_examples(
                    train_rows,
                    teacher_by_id,
                    False,
                    upsample_families=up_fams,
                    upsample_factor=max(1, args.upsample_factor),
                    option_shuffle=True,
                    rng=rng,
                )
            epoch += 1
            order = list(range(len(examples)))
            rng.shuffle(order)
            for start in range(0, len(order), mb):
                if steps_done >= args.steps:
                    break
                idxs = order[start : start + mb]
                batch = [examples[i] for i in idxs if examples[i]["id"] not in hold_set]
                if not batch:
                    continue
                prompts = [b["prompt"] for b in batch]
                opt_offsets_batch = [b["opt_offsets"] for b in batch]
                n_opts_list = [b["n_opts"] for b in batch]

                row_logits = _forward_option_logits(
                    model,
                    tok,
                    device,
                    letter_ids,
                    prompts,
                    opt_offsets_batch,
                    n_opts_list,
                    args.max_length,
                    args.readout,
                    head_module,
                    bidirectional,
                    amp_kind,
                    use_amp,
                )

                loss = torch.zeros((), device=device, dtype=torch.float32)
                for i, ex in enumerate(batch):
                    n = ex["n_opts"]
                    opt_logits = row_logits[i].float()  # explicit fp32 head cast
                    gold = int(ex["label"])
                    ex_weight = float(ex.get("weight", 1.0)) if args.rlsf_weights else 1.0
                    if args.mode == "caps":
                        primary = caps_loss_torch(
                            opt_logits,
                            gold,
                            outcome=float(ex.get("outcome", 1.0)),
                            rule=args.caps_rule,
                        )
                    else:
                        primary = torch.nn.functional.cross_entropy(
                            opt_logits.unsqueeze(0),
                            torch.tensor([gold], device=device),
                        )
                    ex_loss = ce_w * primary
                    soft = ex.get("teacher_probs")
                    if soft is not None and len(soft) == n and kl_w > 0:
                        teacher = torch.tensor(soft, device=device, dtype=torch.float32)
                        teacher = teacher / teacher.sum().clamp_min(1e-8)
                        # KL(teacher || student) with temperature on student logits.
                        log_s = torch.nn.functional.log_softmax(opt_logits / T, dim=0)
                        kl = torch.nn.functional.kl_div(
                            log_s, teacher, reduction="sum"
                        ) * (T * T)
                        ex_loss = ex_loss + kl_w * kl
                    loss = loss + ex_weight * ex_loss
                loss = loss / max(1, len(batch))
                opt.zero_grad()
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                steps_done += 1
                if steps_done == 1 or steps_done % max(1, args.steps // 5) == 0:
                    print(
                        f"step {steps_done}/{args.steps} mode={args.mode} "
                        f"readout={args.readout} loss={loss.item():.4f}"
                    )

        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        if args.readout == "cosine" and head_module is not None:
            ehead_path = head_module.save_ehead(args.out / "cosine_head.ehead")
            meta["ehead"] = str(ehead_path)
            print("wrote cosine head", ehead_path)
        elif args.readout == "clef" and head_module is not None:
            clef_path = args.out / "clef_head.pt"
            torch.save(
                {
                    "readout": "clef",
                    "python_only_ablation": True,
                    "hidden": head_module.hidden,
                    "proj_dim": head_module.proj_dim,
                    "state_dict": head_module.state_dict(),
                },
                clef_path,
            )
            meta["clef_head"] = str(clef_path)
            print("wrote python-only CLEF ablation head", clef_path)
        meta["steps_done"] = steps_done
        meta["epochs_started"] = epoch
        meta["status"] = "trained"
    except Exception as e:
        meta["status"] = "blocked"
        meta["blocked_reason"] = str(e)
        print("train blocked:", e)

    # Holdout eval — student adapter + teacher soft baseline.
    if hold_rows and meta.get("status") == "trained":
        y_t, y_p, confs = _score_batch(
            model,
            tok,
            device,
            letter_ids,
            hold_rows,
            max_length=args.max_length,
            readout=args.readout,
            head_module=head_module,
            bidirectional=bidirectional,
            amp_kind=amp_kind,
            use_amp=use_amp,
        )
        meta["student_holdout"] = _ba_ece_arrays(y_t, y_p, confs)
        # Teacher soft baseline on same holdout IDs.
        ty, tp, tc = [], [], []
        for h in hold_rows:
            t = teacher_by_id.get(h["id"])
            if not t:
                continue
            probs = t.get("probabilities") or t.get("teacher_probs") or []
            if not probs:
                continue
            pred = int(max(range(len(probs)), key=lambda i: float(probs[i])))
            ty.append(int(h["label"]))
            tp.append(pred)
            tc.append(float(probs[pred]))
        if ty:
            meta["teacher_holdout"] = _ba_ece_arrays(ty, tp, tc)
        ba = (meta.get("student_holdout") or {}).get("BA")
        meta["gate_pass"] = bool(ba is not None and ba >= 0.75)
        meta["holdout_note"] = (
            "strict holdout IDs never trained; "
            "unadapted_holdout = base student readout; "
            "student_holdout = adapted student readout; "
            "teacher_holdout = teacher soft argmax; BA/ECE only"
        )

    (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", args.out / "train_status.json")
    keys = (
        "status",
        "mode",
        "readout",
        "attention",
        "amp_resolved",
        "steps_done",
        "n_train",
        "n_holdout",
        "n_examples_aug",
        "ehead",
        "clef_head",
        "unadapted_holdout",
        "teacher_holdout",
        "student_holdout",
        "gate_pass",
        "blocked_reason",
    )
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=2))


if __name__ == "__main__":
    main()

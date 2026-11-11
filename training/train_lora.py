#!/usr/bin/env python3
"""FP32 LoRA distillation for Qwen3-0.6B. Pascal (sm_61) friendly.

Strict train/holdout IDs (no leakage). Gold option-restricted CE is primary;
optional teacher soft KL over letter logits. Cyclic option permutation
augmentation. Batched DOD tensors. Evidence: BA/ECE on holdout only —
never 1e-6, never authored144.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
        hold_lineages: set[str] = set()
        # At least one lineage per family into holdout when possible.
        seen_fam: set[str] = set()
        for lid in lineage_ids:
            fam = str(by_lineage[lid][0].get("family") or "unknown")
            if fam not in seen_fam:
                hold_lineages.add(lid)
                seen_fam.add(fam)
        n_hold = max(len(hold_lineages), int(round(len(lineage_ids) * holdout_frac)))
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


def _prompt_from_row(row: dict) -> str:
    """Render student prompt; default SemIf matches Rust / GGUF scorer.

    Evidence is truncated to ``EVIDENCE_TOKEN_CAP`` *before* options so
    ``max_length=768`` never clips option letters on Pascal FP32.
    """
    state = truncate_evidence(row.get("state", ""), EVIDENCE_TOKEN_CAP)
    if _PROMPT_STYLE == "compact":
        lines = [
            f"Question: {row.get('question', '')}",
            f"State: {state}",
            "Options:",
        ]
        for i, o in enumerate(row.get("options") or []):
            desc = o.get("description") or o.get("id") or ""
            lines.append(f"{LETTERS[i]}. {desc}")
        lines.append("Answer:")
        return "\n".join(lines)
    options = []
    for i, o in enumerate(row.get("options") or []):
        options.append(
            {
                "letter": LETTERS[i],
                "description": o.get("description") or o.get("id") or "",
            }
        )
    payload = {
        "evidence": state,
        "criterion": row.get("question", ""),
        "options": options,
    }
    system = DIRECT_SYSTEM.strip()
    user = _dumps_pythonish(payload).strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _permute_row(row: dict, k: int, teacher_probs: list[float] | None) -> tuple[dict, list[float] | None]:
    """Cyclic rotate options by k; remap gold label and teacher probs."""
    opts = list(row.get("options") or [])
    n = len(opts)
    if n < 2:
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


def _build_examples(
    train_rows: list[dict],
    teacher_by_id: dict,
    permute: bool,
    upsample_families: set[str] | None = None,
    upsample_factor: int = 2,
) -> list[dict]:
    ex = []
    up = upsample_families or set()
    for row in train_rows:
        tid = row["id"]
        soft = None
        t = teacher_by_id.get(tid)
        if t:
            soft = t.get("probabilities") or t.get("teacher_probs")
            if soft is not None:
                soft = [float(x) for x in soft]
        n = len(row.get("options") or [])
        cycles = list(range(n)) if permute and n >= 2 else [0]
        fam = str(row.get("family") or "")
        reps = upsample_factor if fam in up else 1
        for _ in range(reps):
            for k in cycles:
                prow, psoft = _permute_row(row, k, soft)
                ex.append(
                    {
                        "id": tid,
                        "prompt": _prompt_from_row(prow),
                        "label": int(prow["label"]),
                        "n_opts": len(prow.get("options") or []),
                        "teacher_probs": psoft,
                    }
                )
    return ex


def _resolve_amp_dtype(torch, device: str, requested: str):
    """Pick the autocast dtype, degrading gracefully where bf16 has no hardware.

    bf16 needs sm_80+ (Ampere). The pinned venv is also used on a Pascal 1080 Ti
    (sm_61), which has no bf16 hardware, so asking for bf16 there would not give
    the tensor rate it asks for. Resolve once, record the choice in
    train_status.json, and keep the loss head in fp32 regardless.
    """
    if requested == "off" or device == "cpu":
        return None, "fp32 (no autocast)" if device != "cpu" else "fp32 (cpu)"
    if requested == "fp16":
        return torch.float16, "fp16"
    # bf16 / auto both want bf16 when it is real hardware.
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16" if requested == "bf16" else "bf16 (auto)"
    except Exception:
        pass
    return torch.float16, "fp16 (bf16 unsupported on this GPU)"


def _score_batch(
    model, tok, device, letter_ids: list[int], rows: list[dict], batch_size: int = 4, amp_dtype=None
):
    """Batched letter-logit scoring. Returns parallel pred/conf/label arrays."""
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confs: list[float] = []
    prompts = [_prompt_from_row(r) for r in rows]
    labels = [int(r["label"]) for r in rows]
    n_opts_list = [len(r.get("options") or []) for r in rows]

    for start in range(0, len(prompts), batch_size):
        chunk_p = prompts[start : start + batch_size]
        chunk_n = n_opts_list[start : start + batch_size]
        chunk_y = labels[start : start + batch_size]
        enc = tok(
            chunk_p,
            return_tensors="pt",
            truncation=True,
            max_length=768,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            # Autocast only the forward matmuls; score the letter logits in
            # fp32 so BA/ECE stay comparable across precisions.
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_dtype is not None and device == "cuda",
            ):
                logits = model(**enc).logits  # [B, T, V]
            logits = logits.float()
            # Last non-pad position per row.
            attn = enc["attention_mask"]
            last = attn.sum(dim=1) - 1  # [B]
            b = logits.size(0)
            gathered = logits[torch.arange(b, device=device), last]  # [B, V]
        for i in range(b):
            n = chunk_n[i]
            scores = gathered[i, letter_ids[:n]]
            probs = _softmax_t(scores, dim=0)
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
        help="Weight CE loss by row rlsf_weight / advantage when present",
    )
    p.add_argument("--out", type=Path, default=Path("packs/qwen3-0.6b-lora"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--amp",
        choices=("auto", "bf16", "fp16", "off"),
        default="auto",
        help=(
            "autocast dtype for fwd/bwd. auto = bf16 when the GPU supports it "
            "else fp16; off = fp32 (old behaviour). Pascal (sm_61) has no bf16 "
            "and resolves to fp16"
        ),
    )
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
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "student": args.student,
        "teacher_jsonl": str(args.teacher_jsonl) if args.teacher_jsonl else None,
        "gold": str(args.gold) if args.gold else None,
        "steps": args.steps,
        "lr": args.lr,
        "ce_weight": args.ce_weight,
        "kl_weight": args.kl_weight,
        "kl_temp": args.kl_temp,
        "permute": args.permute,
        "note": (
            "strict split; SemIf direct-options-v1 prompts (Rust-parity); "
            "gold CE primary + soft KL; BA/ECE holdout only; never 1e-6/authored144"
        ),
        "prompt_version": (
            "direct-options-v1" if args.prompt_style == "semif" else "compact-student-v1"
        ),
        "prompt_style": args.prompt_style,
        "status": "scaffolded",
        "gate_BA": 0.75,
    }
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
    if torch.cuda.is_available() and not args.cpu:
        meta["gpu"] = torch.cuda.get_device_name(0)

    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=targets)
    meta["lora"] = {"r": args.lora_r, "alpha": args.lora_alpha, "targets": targets}

    if args.steps <= 0:
        meta["status"] = "smoke"
        (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("smoke only", meta["torch"], meta["cuda"])
        return

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    meta["device"] = device
    amp_dtype, amp_label = _resolve_amp_dtype(torch, device, args.amp)
    meta["amp_requested"] = args.amp
    meta["amp_resolved"] = amp_label
    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = _letter_id_cache(tok, 16)

    # Weights in the autocast dtype so the matmuls run on the tensor path;
    # the letter-logit loss head is cast back to fp32 at use.
    weight_dtype = amp_dtype if amp_dtype is not None else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=weight_dtype, trust_remote_code=True
    )
    base.to(device)

    # Unadapted student baseline on holdout (same prompt as student).
    if hold_rows:
        y_t, y_p, confs = _score_batch(
            base, tok, device, letter_ids, hold_rows, amp_dtype=amp_dtype
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
    model.train()
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=args.lr)

    up_fams = {f.strip() for f in args.upsample_families.split(",") if f.strip()}
    examples = _build_examples(
        train_rows,
        teacher_by_id,
        args.permute,
        upsample_families=up_fams,
        upsample_factor=max(1, args.upsample_factor),
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

    rng = random.Random(args.split_seed)
    mb = max(1, args.microbatch)
    ce_w, kl_w, T = args.ce_weight, args.kl_weight, args.kl_temp
    steps_done = 0
    order = list(range(len(examples)))
    # fp16 needs loss scaling to avoid underflow; bf16 has fp32's exponent
    # range and must NOT be scaled.
    use_scaler = amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None
    use_amp = amp_dtype is not None and device == "cuda"
    meta["grad_scaler"] = bool(use_scaler)
    # Throughput evidence: steps/s over the timed region, so the bf16 win is
    # measurable rather than asserted.
    _t_start = None
    _t_steps = 0

    try:
        while steps_done < args.steps:
            rng.shuffle(order)
            for start in range(0, len(order), mb):
                if steps_done >= args.steps:
                    break
                idxs = order[start : start + mb]
                batch = [examples[i] for i in idxs]
                # Skip any example whose source id is holdout (belt+suspenders).
                batch = [b for b in batch if b["id"] not in set(hold_ids)]
                if not batch:
                    continue
                prompts = [b["prompt"] for b in batch]
                enc = tok(
                    prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                    padding=True,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                # Autocast the transformer matmuls; the letter-logit head below
                # stays fp32 so CE/KL and the .item() loss print are unchanged.
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    out = model(**enc)
                    logits = out.logits
                logits = logits.float()
                attn = enc["attention_mask"]
                last = attn.sum(dim=1) - 1
                bsz = logits.size(0)
                gathered = logits[torch.arange(bsz, device=device), last]  # [B, V]

                loss = torch.zeros((), device=device)
                for i, ex in enumerate(batch):
                    n = ex["n_opts"]
                    letter_logits = gathered[i, letter_ids[:n]]  # [n]
                    gold = int(ex["label"])
                    ce = torch.nn.functional.cross_entropy(
                        letter_logits.unsqueeze(0),
                        torch.tensor([gold], device=device),
                    )
                    loss = loss + ce_w * ce
                    soft = ex.get("teacher_probs")
                    if soft is not None and len(soft) == n and kl_w > 0:
                        teacher = torch.tensor(soft, device=device, dtype=torch.float32)
                        teacher = teacher / teacher.sum().clamp_min(1e-8)
                        # KL(teacher || student) with temperature on student logits.
                        log_s = torch.nn.functional.log_softmax(letter_logits / T, dim=0)
                        kl = torch.nn.functional.kl_div(
                            log_s, teacher, reduction="sum"
                        ) * (T * T)
                        loss = loss + kl_w * kl
                loss = loss / max(1, len(batch))
                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                steps_done += 1
                # Start the clock after step 1 so CUDA warmup/cuBLAS autotune
                # does not flatter the first precision measured.
                if steps_done == 2:
                    _t_start = time.perf_counter()
                    _t_steps = 0
                elif _t_start is not None:
                    _t_steps += 1
                if steps_done == 1 or steps_done % max(1, args.steps // 5) == 0:
                    print(f"step {steps_done}/{args.steps} loss={loss.item():.4f}")

        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        meta["steps_done"] = steps_done
        if _t_start is not None and _t_steps > 0:
            elapsed = time.perf_counter() - _t_start
            meta["train_secs"] = round(elapsed, 3)
            meta["timed_steps"] = _t_steps
            meta["steps_per_sec"] = round(_t_steps / elapsed, 3)
        meta["status"] = "trained"
    except Exception as e:
        meta["status"] = "blocked"
        meta["blocked_reason"] = str(e)
        print("train blocked:", e)

    # Holdout eval — student adapter + teacher soft baseline.
    if hold_rows and meta.get("status") == "trained":
        y_t, y_p, confs = _score_batch(
            model, tok, device, letter_ids, hold_rows, amp_dtype=amp_dtype
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
            "unadapted_holdout = base student letter logits; "
            "student_holdout = adapter letter logits; "
            "teacher_holdout = teacher soft argmax; BA/ECE only"
        )

    (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", args.out / "train_status.json")
    keys = (
        "status",
        "steps_done",
        "n_train",
        "n_holdout",
        "n_examples_aug",
        "unadapted_holdout",
        "teacher_holdout",
        "student_holdout",
        "gate_pass",
        "blocked_reason",
    )
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=2))


if __name__ == "__main__":
    main()

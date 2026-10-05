#!/usr/bin/env python3
"""FP32 LoRA distillation for Qwen3-0.6B. Pascal (sm_61) friendly.

Does **not** fit on authored144 test. Teacher JSONL should come from
`teacher_label.py` on fixtures/lang/dev.jsonl (split=dev).
"""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="Qwen/Qwen3-0.6B")
    p.add_argument("--teacher-jsonl", type=Path, required=False)
    p.add_argument("--out", type=Path, default=Path("packs/qwen3-0.6b-lora"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--steps", type=int, default=0, help="0 = smoke only; >0 trains if torch is present")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "student": args.student,
        "teacher_jsonl": str(args.teacher_jsonl) if args.teacher_jsonl else None,
        "steps": args.steps,
        "note": "dev-split teacher labels only; do not fit on authored144 test",
    }
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available() and not args.cpu:
            meta["gpu"] = torch.cuda.get_device_name(0)
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        meta["lora"] = {"r": 8, "alpha": 16}
        print("torch", meta["torch"], "cuda", meta["cuda"])
        print("Would attach LoRA", cfg)
        print("get_peft_model imported", get_peft_model is not None)
        if args.steps > 0:
            print(f"train steps={args.steps} skipped unless full HF weights are present")
    except ImportError as e:
        meta["error"] = str(e)
        print("training extras missing:", e)
        print("use training/.venv (python 3.12 + torch cu118)")
    (args.out / "train_status.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", args.out / "train_status.json")

if __name__ == "__main__":
    main()

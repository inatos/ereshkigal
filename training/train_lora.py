#!/usr/bin/env python3
"""FP32 LoRA distillation for Qwen3-0.6B. Pascal (sm_61) friendly."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="Qwen/Qwen3-0.6B")
    p.add_argument("--teacher-jsonl", type=Path, required=False)
    p.add_argument("--out", type=Path, default=Path("packs/qwen3-0.6b-lora"))
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        if torch.cuda.is_available() and not args.cpu:
            print("gpu", torch.cuda.get_device_name(0), "arch", torch.cuda.get_arch_list())
        print("Would attach LoRA", LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
        print("get_peft_model imported", get_peft_model is not None)
        print("write adapters under", args.out)
    except ImportError as e:
        print("training extras missing:", e)
        print("use training/.venv (python 3.12 + torch cu118)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from diffusers import DiffusionPipeline, Transformer2DModel
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with finetuned PixArt LoRA on a new subset of Gundam inputs"
    )
    p.add_argument(
        "--base_combined",
        type=str,
        default="PixArt-sigma/output/pretrained_models/pixart_sigma_combined",
        help="Path to combined base",
    )
    p.add_argument(
        "--lora_dir",
        type=str,
        default="PixArt-sigma/output/gundam_lora_512",
        help="Path to LoRA directory",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="local_repo/PixArt_ft/output_new",
        help="Output directory for generated images",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of new Gundam samples to run",
    )
    p.add_argument(
        "--skip_first_n",
        type=int,
        default=1028,
        help="Skip the first N samples to avoid using training subset (adjust to your training count)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    transformer = Transformer2DModel.from_pretrained(
        args.base_combined, subfolder="transformer", torch_dtype=dtype
    )
    transformer = PeftModel.from_pretrained(transformer, args.lora_dir)
    pipe = DiffusionPipeline.from_pretrained(
        args.base_combined, transformer=transformer, torch_dtype=dtype
    )
    pipe = pipe.to(args.device)

    ds = load_dataset("Gazoche/gundam-captioned", split="train")
    start = min(args.skip_first_n, len(ds))
    end = min(start + args.num_samples, len(ds))

    for i in range(start, end):
        cap = ds[i]["text"]
        img = pipe(cap, height=512, width=512, num_inference_steps=20).images[0]
        img.save(out / f"{i:05d}.png")
        (out / f"{i:05d}_caption.txt").write_text(cap)

    print(f"Generated {end - start} images to {out}")


if __name__ == "__main__":
    main()

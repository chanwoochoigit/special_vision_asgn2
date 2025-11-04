#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, Transformer2DModel
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate finetuned PixArt LoRA by loading pipeline and generating a test image"
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
        default="PixArt-sigma/output/wikiart_lora_512",
        help="Path to LoRA output directory",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="local_repo/WikiArt_ft/validation",
        help="Directory to save validation image",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Artist: berthe-morisot, Style: Impressionism, Genre: landscape",
        help="Validation prompt",
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
    base = Path(args.base_combined)
    lora = Path(args.lora_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    transformer = Transformer2DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=dtype
    )
    transformer = PeftModel.from_pretrained(transformer, lora)
    pipe = DiffusionPipeline.from_pretrained(
        base, transformer=transformer, torch_dtype=dtype
    )
    pipe = pipe.to(args.device)

    image = pipe(args.prompt, height=512, width=512, num_inference_steps=20).images[0]
    out_path = out / "validation_sample.png"
    image.save(out_path)
    print(f"Validation OK. Sample saved to: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
from pathlib import Path
import time
import json

import torch
from diffusers import DiffusionPipeline, Transformer2DModel
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference with Gundam LoRA using existing caption files"
    )
    p.add_argument(
        "--base_combined",
        type=str,
        default="PixArt-sigma/output/pretrained_models/pixart_sigma_combined",
    )
    p.add_argument(
        "--lora_dir", type=str, default="PixArt-sigma/output/gundam_lora_512"
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default="local_repo/PixArt/input/test",
        help="Directory containing the _caption.txt files",
    )
    p.add_argument("--out_dir", type=str, default="local_repo/PixArt_ft/output")
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--cfg", type=float, default=4.5)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    transformer = Transformer2DModel.from_pretrained(
        args.base_combined, subfolder="transformer", torch_dtype=dtype
    )
    transformer = PeftModel.from_pretrained(transformer, args.lora_dir)
    pipe = DiffusionPipeline.from_pretrained(
        args.base_combined, transformer=transformer, torch_dtype=dtype
    )
    pipe = pipe.to(args.device)

    # Find all caption files in the input directory and sort them
    caption_files = sorted(input_path.glob("*_caption.txt"))

    if not caption_files:
        print(f"No *_caption.txt files found in {input_path}")
        return

    print(f"Found {len(caption_files)} captions to process from {input_path}")

    timings = []
    # Loop over the found caption files
    for caption_file in caption_files:
        # e.g., "01800_caption"
        base_stem = caption_file.stem
        # e.g., "01800"
        base_name = base_stem.replace("_caption", "")

        # Read the caption directly from the file
        cap = caption_file.read_text().strip()

        t0 = time.time()
        img = pipe(
            cap, height=512, width=512, num_inference_steps=20, guidance_scale=args.cfg
        ).images[0]
        t1 = time.time()

        # Save using the base name from the file
        img.save(out / f"{base_name}.png")
        (out / f"{base_name}_caption.txt").write_text(cap)
        timings.append({"file": base_name, "time_s": float(t1 - t0)})

    with (out / "timings.json").open("w") as f:
        json.dump(timings, f, indent=2)

    print(f"Generated {len(timings)} images to {out}. Timings saved to timings.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
from pathlib import Path
import time
import json

import torch
from datasets import load_dataset
from PIL import Image
from diffusers import DiffusionPipeline, Transformer2DModel
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference with WikiArt LoRA on a new subset"
    )
    p.add_argument(
        "--base_combined",
        type=str,
        default="PixArt-sigma/output/pretrained_models/pixart_sigma_combined",
    )
    p.add_argument(
        "--lora_dir", type=str, default="PixArt-sigma/output/wikiart_lora_512"
    )
    p.add_argument("--out_dir", type=str, default="local_repo/WikiArt_ft/output_new")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--skip_first_n", type=int, default=0)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--prompt_field", type=str, default="style")
    p.add_argument(
        "--use_meta",
        action="store_true",
        help="Use saved meta.json to compose prompts if available",
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

    ds = load_dataset("huggan/wikiart", split="train")
    start = min(args.skip_first_n, len(ds))
    end = min(start + args.num_samples, len(ds))

    timings = []
    for i in range(start, end):
        # Prefer meta.json composition if requested
        cap = None
        if args.use_meta:
            meta_path = Path("local_repo/WikiArt/input/train") / f"{i:05d}_meta.json"
            if meta_path.exists():
                import json as _json

                meta = _json.loads(meta_path.read_text())
                parts = []
                title = str(meta.get("title", "")).strip()
                style = str(meta.get("style", "")).strip()
                genre = str(meta.get("genre", "")).strip()
                artist = (
                    str(meta.get("artist", "")).strip()
                    or str(meta.get("artist_name", "")).strip()
                )
                if title:
                    parts.append(title)
                if style:
                    parts.append(f"in {style} style")
                if genre:
                    parts.append(genre)
                if artist:
                    parts.append(f"by {artist}")
                cap = ", ".join([p for p in parts if p]) or None
        if not cap:
            cap = str(ds[i].get(args.prompt_field, "")).strip() or "a painting"
        t0 = time.time()
        img = pipe(
            cap, height=512, width=512, num_inference_steps=20, guidance_scale=args.cfg
        ).images[0]
        t1 = time.time()
        img.save(out / f"{i:05d}.png")
        (out / f"{i:05d}_caption.txt").write_text(cap)
        timings.append({"index": i, "time_s": float(t1 - t0)})

    with (out / "timings.json").open("w") as f:
        json.dump(timings, f, indent=2)
    print(f"Generated {end - start} images to {out}. Timings saved to timings.json")


if __name__ == "__main__":
    main()

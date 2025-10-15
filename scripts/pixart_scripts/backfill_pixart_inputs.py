#!/usr/bin/env python3
import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill PixArt input images and captions based on existing outputs."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="local_repo/PixArt",
        help="Base directory containing 'output' and 'input' subfolders.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Gazoche/gundam-captioned",
        help="Hugging Face dataset path to load images and captions from.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size to which input images will be resized before saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing input files if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base = Path(args.base_dir)
    output_dir = base / "output"
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    outputs = sorted(output_dir.glob("*.png"))
    if not outputs:
        print(f"No outputs found in: {output_dir}")
        return

    print(f"Found {len(outputs)} output images. Loading dataset '{args.dataset}'…")
    ds = load_dataset(args.dataset, split=args.split)
    total = 0
    written = 0

    for out_path in outputs:
        stem = out_path.stem  # e.g. 00012
        try:
            idx = int(stem)
        except ValueError:
            continue

        if idx >= len(ds):
            continue

        img = ds[idx]["image"].convert("RGB").resize((args.image_size, args.image_size))
        cap = str(ds[idx]["text"]) if "text" in ds.features else ""

        img_out = input_dir / f"{stem}_input.png"
        cap_out = input_dir / f"{stem}_caption.txt"

        if args.overwrite or not img_out.exists():
            img.save(img_out)
            written += 1
        if args.overwrite or not cap_out.exists():
            cap_out.write_text(cap)
            written += 1
        total += 1

    print(f"Processed {total} outputs; wrote {written} input files into {input_dir}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill PixArt input images and captions based on existing outputs."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="local_repo/PixArt",
        help="Base directory containing 'output' and 'input' subfolders.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Gazoche/gundam-captioned",
        help="Hugging Face dataset path to load images and captions from.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size to which input images will be resized before saving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing input files if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base = Path(args.base_dir)
    output_dir = base / "output"
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    outputs = sorted(output_dir.glob("*.png"))
    if not outputs:
        print(f"No outputs found in: {output_dir}")
        return

    print(f"Found {len(outputs)} output images. Loading dataset '{args.dataset}'…")
    ds = load_dataset(args.dataset, split=args.split)
    total = 0
    written = 0

    for out_path in outputs:
        stem = out_path.stem  # e.g. 00012
        try:
            idx = int(stem)
        except ValueError:
            # Skip files that don't follow zero-padded integer naming
            continue

        if idx >= len(ds):
            continue

        img = ds[idx]["image"].convert("RGB").resize((args.image_size, args.image_size))
        cap = str(ds[idx]["text"]) if "text" in ds.features else ""

        img_out = input_dir / f"{stem}_input.png"
        cap_out = input_dir / f"{stem}_caption.txt"

        if args.overwrite or not img_out.exists():
            img.save(img_out)
            written += 1
        if args.overwrite or not cap_out.exists():
            cap_out.write_text(cap)
            written += 1
        total += 1

    print(f"Processed {total} outputs; wrote {written} input files into {input_dir}.")


if __name__ == "__main__":
    main()

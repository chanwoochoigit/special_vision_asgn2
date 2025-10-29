#!/usr/bin/env python
import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare PixArt inputs (images+captions) directly from WikiArt dataset"
    )
    p.add_argument("--base_dir", type=str, default="local_repo/WikiArt")
    p.add_argument("--dataset", type=str, default="huggan/wikiart")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument(
        "--caption_field", type=str, default="style", help="Field to use as caption"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_dir)
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.shuffle(seed=args.seed)
    n_total = min(args.num_samples, len(ds))
    n_train = int(n_total * args.train_ratio)
    train_ds = ds.select(range(n_train))
    test_ds = ds.select(range(n_train, n_total))

    def write_split(split_ds, split_name):
        subdir = input_dir / split_name
        subdir.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(split_ds):
            img = rec["image"].convert("RGB").resize((args.image_size, args.image_size))
            # prefer text-like fields
            cap = ""
            for key in ["title", "genre", "style", args.caption_field]:
                if key in rec and isinstance(rec[key], str) and rec[key].strip():
                    cap = rec[key].strip()
                    break
            if not cap:
                cap = str(rec.get(args.caption_field, "painting")).strip() or "painting"
            img.save(subdir / f"{i:05d}_input.png")
            (subdir / f"{i:05d}_caption.txt").write_text(cap)

            # Save feature metadata for later use (artist/genre/style/etc.)
            meta = {}
            for k, v in rec.items():
                if k == "image":
                    continue
                # Convert to basic python types where possible
                if hasattr(v, "item"):
                    v = v.item()

                # stringify anything not JSON-serializable by default
                try:
                    import json as _json

                    _json.dumps(v)
                    meta[k] = v
                except Exception:
                    meta[k] = str(v)
            (subdir / f"{i:05d}_meta.json").write_text(
                __import__("json").dumps(meta, ensure_ascii=False, indent=2)
            )

    write_split(train_ds, "train")
    write_split(test_ds, "test")

    print(
        f"Prepared {len(train_ds)} train and {len(test_ds)} test inputs in {input_dir}"
    )


if __name__ == "__main__":
    main()

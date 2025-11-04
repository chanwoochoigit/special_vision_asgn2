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

    def write_split(split_ds, split_name, start_index=0):
        subdir = input_dir / split_name
        subdir.mkdir(parents=True, exist_ok=True)
        artist_names = split_ds.features["artist"].names
        style_names = split_ds.features["style"].names
        genre_names = split_ds.features["genre"].names

        for i, rec in enumerate(split_ds, start=start_index):
            img = rec["image"].convert("RGB").resize((args.image_size, args.image_size))
            caption_parts = []
            artist_int = rec.get("artist")
            if artist_int is not None:
                artist_str = artist_names[artist_int]
                caption_parts.append(f"Artist: {artist_str.replace('_', ' ')}")

            style_int = rec.get("style")
            if style_int is not None:
                style_str = style_names[style_int]
                caption_parts.append(f"Style: {style_str.replace('_', ' ')}")

            genre_int = rec.get("genre")
            if genre_int is not None:
                genre_str = genre_names[genre_int]
                caption_parts.append(f"Genre: {genre_str.replace('_', ' ')}")

            cap = ", ".join(caption_parts)
            img.save(subdir / f"{i:05d}_input.png")
            (subdir / f"{i:05d}_caption.txt").write_text(cap)

    write_split(train_ds, "train", start_index=0)
    write_split(test_ds, "test", start_index=len(train_ds))

    print(
        f"Prepared {len(train_ds)} train and {len(test_ds)} test inputs in {input_dir}"
    )


if __name__ == "__main__":
    main()

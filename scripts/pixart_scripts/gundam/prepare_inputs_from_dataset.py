#!/usr/bin/env python
import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare PixArt inputs (images+captions) from Gundam dataset"
    )
    p.add_argument(
        "--base_dir",
        type=str,
        default="local_repo/PixArt",
        help="Base directory for the 'input' subfolder.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="Gazoche/gundam-captioned",
        help="Hugging Face dataset path.",
    )
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Total number of samples to pull from the dataset.",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size to resize images to.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for shuffling to ensure deterministic split.",
    )
    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of samples to use for training.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_dir)
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{args.dataset}'...")
    ds = load_dataset(args.dataset, split=args.split)

    print(f"Shuffling dataset with seed {args.seed}...")
    ds = ds.shuffle(seed=args.seed)

    # Calculate splits
    n_total = min(args.num_samples, len(ds))
    n_train = int(n_total * args.train_ratio)
    train_ds = ds.select(range(n_train))
    test_ds = ds.select(range(n_train, n_total))

    print(
        f"Total samples: {n_total}. Creating {n_train} train and {len(test_ds)} test samples."
    )

    def write_split(split_ds, split_name, start_index=0):
        """Saves images and captions for a given split."""
        subdir = input_dir / split_name
        subdir.mkdir(parents=True, exist_ok=True)

        print(
            f"Writing {len(split_ds)} samples to {subdir} starting from index {start_index}..."
        )

        for i, rec in enumerate(split_ds, start=start_index):
            # Get image and resize
            img = rec["image"].convert("RGB").resize((args.image_size, args.image_size))

            # Get caption from the 'text' field
            cap = str(rec.get("text", ""))

            # Save files with global index
            img_path = subdir / f"{i:05d}_input.png"
            cap_path = subdir / f"{i:05d}_caption.txt"

            img.save(img_path)
            cap_path.write_text(cap)

    # Write the train split (starts from index 0)
    write_split(train_ds, "train", start_index=0)

    # Write the test split (starts from index n_train)
    write_split(test_ds, "test", start_index=n_train)

    print(
        f"\nDone. Prepared {len(train_ds)} train and {len(test_ds)} test inputs in {input_dir}"
    )


if __name__ == "__main__":
    main()

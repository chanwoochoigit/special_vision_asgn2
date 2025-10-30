#!/usr/bin/env python
import argparse
from pathlib import Path
import json


def parse_args():
    p = argparse.ArgumentParser(
        description="Create metadata.jsonl for HF imagefolder from {index}_input + {index}_caption pairs"
    )
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--metadata_name", type=str, default="metadata.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    meta_path = in_dir / args.metadata_name

    records = []
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for img_path in in_dir.glob(f"*{ext}"):
            stem = img_path.stem
            txt_path = img_path.with_suffix(".txt")

            cap = ""
            if txt_path.exists():
                cap = txt_path.read_text(encoding="utf-8", errors="ignore").strip()

            # Ensure the filename stem is a parsable integer (like '00000')
            try:
                int(stem)
                records.append({"file_name": img_path.name, "text": cap, "stem": stem})
            except ValueError:
                # Not an image with an integer stem, skip it
                print(f"Skipping non-standard file: {img_path.name}")
                continue

    # Sort records numerically by the stem (e.g., '00000', '00001', ...)
    records.sort(key=lambda r: int(r["stem"]))

    count = 0
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in records:
            # Write the final record without the temporary 'stem' key
            final_rec = {"file_name": rec["file_name"], "text": rec["text"]}
            f.write(json.dumps(final_rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {meta_path}")


if __name__ == "__main__":
    main()

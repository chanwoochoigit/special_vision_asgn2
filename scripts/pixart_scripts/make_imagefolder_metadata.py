#!/usr/bin/env python3
import argparse
from pathlib import Path
import json


def parse_args():
    p = argparse.ArgumentParser(
        description="Create metadata.jsonl for HF imagefolder from PixArt/input pairs"
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default="local_repo/PixArt/input",
        help="Folder containing {index}_input.(png|jpg|jpeg|webp) and {index}_caption.txt",
    )
    p.add_argument(
        "--metadata_name",
        type=str,
        default="metadata.jsonl",
        help="Output metadata filename (written into input_dir)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    meta_path = in_dir / args.metadata_name

    captions = {}
    for cap in in_dir.glob("*_caption.txt"):
        stem = cap.stem
        idx = stem.split("_caption")[0]
        try:
            int(idx)
        except ValueError:
            continue
        captions[idx] = cap.read_text(encoding="utf-8", errors="ignore").strip()

    candidates = {}
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for img in in_dir.glob(f"*_input{ext}"):
            stem = img.stem
            idx = stem.split("_input")[0]
            try:
                int(idx)
            except ValueError:
                continue
            candidates[idx] = img.name

    count = 0
    with meta_path.open("w", encoding="utf-8") as f:
        for idx in sorted(candidates.keys(), key=lambda x: int(x)):
            img_name = candidates[idx]
            cap = captions.get(idx, "")
            rec = {"file_name": img_name, "text": cap}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {meta_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
import json


def parse_args():
    p = argparse.ArgumentParser(
        description="Create metadata.jsonl for HF imagefolder from PixArt/input pairs"
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default="local_repo/PixArt/input",
        help="Folder containing {index}_input.(png|jpg|jpeg|webp) and {index}_caption.txt",
    )
    p.add_argument(
        "--metadata_name",
        type=str,
        default="metadata.jsonl",
        help="Output metadata filename (written into input_dir)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    meta_path = in_dir / args.metadata_name

    # Build mapping from index -> caption and image
    captions = {}
    for cap in in_dir.glob("*_caption.txt"):
        stem = cap.stem  # e.g., 00000_caption
        idx = stem.split("_caption")[0]
        try:
            int(idx)
        except ValueError:
            continue
        captions[idx] = cap.read_text(encoding="utf-8", errors="ignore").strip()

    candidates = {}
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for img in in_dir.glob(f"*_input{ext}"):
            stem = img.stem  # e.g., 00000_input
            idx = stem.split("_input")[0]
            try:
                int(idx)
            except ValueError:
                continue
            candidates[idx] = img.name  # relative filename

    count = 0
    with meta_path.open("w", encoding="utf-8") as f:
        for idx in sorted(candidates.keys(), key=lambda x: int(x)):
            img_name = candidates[idx]
            cap = captions.get(idx, "")
            # HF imagefolder expects either {"file_name": ..., "text": ...} or {"image": ..., "text": ...}
            rec = {"file_name": img_name, "text": cap}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {meta_path}")


if __name__ == "__main__":
    main()

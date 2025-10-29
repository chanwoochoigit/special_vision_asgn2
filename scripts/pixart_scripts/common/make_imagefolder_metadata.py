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

    captions = {}
    for cap in in_dir.glob("*_caption.txt"):
        stem = cap.stem
        idx = stem.split("_caption")[0]
        base_cap = cap.read_text(encoding="utf-8", errors="ignore").strip()
        # If a metadata json exists, compose a richer caption using common fields
        meta_path = in_dir / f"{idx}_meta.json"
        if meta_path.exists():
            import json as _json

            meta = _json.loads(meta_path.read_text())
            parts = []
            artist = (
                str(meta.get("artist", "")).strip()
                or str(meta.get("artist_name", "")).strip()
            )
            style = str(meta.get("style", "")).strip()
            genre = str(meta.get("genre", "")).strip()
            title = str(meta.get("title", "")).strip()
            year = str(meta.get("date", "")).strip()
            if title:
                parts.append(title)
            if style:
                parts.append(f"in {style} style")
            if genre:
                parts.append(genre)
            if artist:
                parts.append(f"by {artist}")
            if year:
                parts.append(year)
            composed = ", ".join([p for p in parts if p])
            captions[idx] = composed or base_cap
        else:
            captions[idx] = base_cap

    candidates = {}
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        for img in in_dir.glob(f"*_input{ext}"):
            stem = img.stem
            idx = stem.split("_input")[0]
            int(idx)
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

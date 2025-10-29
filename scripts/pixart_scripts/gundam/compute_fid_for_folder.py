#!/usr/bin/env python
import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from scipy.linalg import sqrtm


def get_inception_features(images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
    )
    inception.fc = nn.Identity()
    inception = inception.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    feats = []
    with torch.no_grad():
        for img in images:
            t = transform(img).unsqueeze(0).to(device)
            feat = inception(t).cpu().numpy()
            feats.append(feat)
    return np.vstack(feats)


def calculate_fid(real_images, generated_images):
    real_features = get_inception_features(real_images)
    gen_features = get_inception_features(generated_images)
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean))
    return fid


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute FID between a Gundam subset and a generated folder"
    )
    p.add_argument(
        "--generated_dir", type=str, required=True, help="Folder with generated images"
    )
    p.add_argument("--dataset", type=str, default="Gazoche/gundam-captioned")
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--start",
        type=int,
        default=1028,
        help="Start index in dataset (to match generated names)",
    )
    p.add_argument("--count", type=int, default=50, help="How many images")
    p.add_argument(
        "--print_timing_stats",
        action="store_true",
        help="If timings.json exists under generated_dir, print avg/min/max/std timing",
    )
    return p.parse_args()


def main():
    args = parse_args()
    gen_dir = Path(args.generated_dir)
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    start = args.start
    end = min(start + args.count, len(ds))

    reals = []
    gens = []
    for i in range(start, end):
        gen_path = gen_dir / f"{i:05d}.png"
        if not gen_path.exists():
            continue
        gens.append(Image.open(gen_path).convert("RGB"))
        reals.append(ds[i]["image"].convert("RGB"))

    fid = calculate_fid(reals, gens)
    print(f"FID: {fid:.4f} (n={len(gens)})")

    if args.print_timing_stats:
        tpath = gen_dir / "timings.json"
        if tpath.exists():
            with tpath.open("r") as f:
                timings = json.load(f)
            vals = [float(t.get("time_s", 0.0)) for t in timings]
            import numpy as _np

            if vals:
                print(
                    f"Timing stats (s) -> avg: {_np.mean(vals):.4f}, std: {_np.std(vals):.4f}, min: {min(vals):.4f}, max: {max(vals):.4f}"
                )


if __name__ == "__main__":
    main()

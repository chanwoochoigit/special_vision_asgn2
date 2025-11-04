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
        description="Compute FID between a folder of real images and a folder of generated images"
    )
    p.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Folder with generated images (e.g., 01800.png)",
    )
    p.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Folder with corresponding real images (e.g., 01800_input.png)",
    )
    p.add_argument("--print_timing_stats", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gen_dir = Path(args.generated_dir)
    real_dir = Path(args.real_dir)

    reals = []
    gens = []

    if not gen_dir.exists():
        print(f"Error: Generated directory not found: {gen_dir}")
        return
    if not real_dir.exists():
        print(f"Error: Real directory not found: {real_dir}")
        return

    # Find all generated images
    generated_files = sorted(gen_dir.glob("*.png"))
    if not generated_files:
        print(f"No .png files found in {gen_dir}")
        return

    print(
        f"Found {len(generated_files)} generated images. Matching with real images..."
    )

    # Loop over generated files and find their real counterparts
    for gen_path in generated_files:
        # e.g., "01800"
        base_name = gen_path.stem
        # e.g., "local_repo/WikiArt/input/test/01800_input.png"
        real_path = real_dir / f"{base_name}_input.png"

        if not real_path.exists():
            print(
                f"Warning: Skipping {gen_path.name}, corresponding real image not found at {real_path}"
            )
            continue

        # Load the pair
        gens.append(Image.open(gen_path).convert("RGB"))
        reals.append(Image.open(real_path).convert("RGB"))

    if not gens:
        print("No matching image pairs were found.")
        return

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

#!/usr/bin/env python
import argparse
from pathlib import Path
import json

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import colour

from loguru import logger

logger.add("logs/metrics.log")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Delta E 2000 between a folder of real images and a folder of generated images"
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


def load_image_as_array(image_path):
    """Load image and convert to numpy array"""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def compute_deltae2000_for_images(real_paths, generated_paths):
    """Compute Delta E 2000 for pairs of images (per-pixel)"""
    all_deltae_values = []

    for img_idx, (real_path, gen_path) in enumerate(zip(real_paths, generated_paths)):
        # Load images as numpy arrays
        real_img = load_image_as_array(real_path)
        gen_img = load_image_as_array(gen_path)

        # Ensure same size (resize gen to match real if needed)
        if real_img.shape != gen_img.shape:
            gen_img_pil = Image.fromarray(gen_img)
            real_img_pil = Image.fromarray(real_img)
            gen_img_pil = gen_img_pil.resize(
                real_img_pil.size, Image.Resampling.LANCZOS
            )
            gen_img = np.array(gen_img_pil)

        # Convert RGB to Lab color space
        # rgb2lab expects float in [0, 1] range
        real_lab = rgb2lab(real_img.astype(np.float64) / 255.0)
        gen_lab = rgb2lab(gen_img.astype(np.float64) / 255.0)

        # Compute Delta E 2000 for all pixels using vectorized operation
        # colour.delta_E uses Lab color arrays directly
        # Reshape to (N, 3) where N is number of pixels
        h, w, c = real_lab.shape
        real_lab_flat = real_lab.reshape(-1, 3)  # (h*w, 3)
        gen_lab_flat = gen_lab.reshape(-1, 3)  # (h*w, 3)

        # Compute Delta E 2000 for all pixels at once (vectorized)
        # colour.delta_E expects arrays of shape (N, 3) or (3,)
        deltae_values = colour.delta_E(real_lab_flat, gen_lab_flat, method="CIE 2000")

        all_deltae_values.extend(deltae_values.tolist())

    return all_deltae_values


def main():
    args = parse_args()
    gen_dir = Path(args.generated_dir)
    real_dir = Path(args.real_dir)

    real_paths = []
    gen_paths = []

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

        gen_paths.append(gen_path)
        real_paths.append(real_path)

    if not gen_paths:
        print("No matching image pairs were found.")
        return

    print("Computing Delta E 2000 for all pixels...")
    all_deltae_values = compute_deltae2000_for_images(real_paths, gen_paths)

    if not all_deltae_values:
        print("No Delta E values computed.")
        return

    # Compute statistics
    mean_deltae = np.mean(all_deltae_values)
    p95_deltae = np.percentile(all_deltae_values, 95)

    logger.info(
        f"Delta E 2000 - Mean: {mean_deltae:.4f}, P95: {p95_deltae:.4f} (n={len(all_deltae_values)} pixels)"
    )

    if args.print_timing_stats:
        tpath = gen_dir / "timings.json"
        if tpath.exists():
            with tpath.open("r") as f:
                timings = json.load(f)
            vals = [float(t.get("time_s", 0.0)) for t in timings]
            import numpy as _np

            if vals:
                logger.info(
                    f"Timing stats (s) -> avg: {_np.mean(vals):.4f}, std: {_np.std(vals):.4f}, min: {min(vals):.4f}, max: {max(vals):.4f}"
                )


if __name__ == "__main__":
    main()

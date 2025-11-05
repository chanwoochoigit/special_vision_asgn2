#!/usr/bin/env python
import argparse
from pathlib import Path
import json

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from loguru import logger

logger.add("logs/metrics.log")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute SSIM between a folder of real images and a folder of generated images"
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
    p.add_argument(
        "--data_range",
        type=float,
        default=255.0,
        help="Data range of the input image (default: 255.0 for [0,255] uint8 images)",
    )
    p.add_argument("--print_timing_stats", action="store_true")
    return p.parse_args()


def load_image_as_array(image_path):
    """Load image and convert to numpy array"""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def compute_ssim_for_images(real_paths, generated_paths, data_range=255.0):
    """Compute SSIM for pairs of images"""
    ssim_values = []

    for real_path, gen_path in zip(real_paths, generated_paths):
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

        # Compute SSIM for RGB images (channel_axis=2 for HxWx3 format)
        ssim_value = ssim(real_img, gen_img, data_range=data_range, channel_axis=2)

        ssim_values.append(ssim_value)

    return ssim_values


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

    ssim_values = compute_ssim_for_images(
        real_paths, gen_paths, data_range=args.data_range
    )

    avg_ssim = sum(ssim_values) / len(ssim_values)
    logger.info(f"SSIM: {avg_ssim:.4f} (n={len(ssim_values)})")

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

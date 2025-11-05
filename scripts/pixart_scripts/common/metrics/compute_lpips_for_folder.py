#!/usr/bin/env python
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import lpips

from loguru import logger

logger.add("logs/metrics.log")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute LPIPS between a folder of real images and a folder of generated images"
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
        "--net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Network to use for LPIPS (default: alex)",
    )
    p.add_argument("--print_timing_stats", action="store_true")
    return p.parse_args()


def load_and_preprocess_image(image_path, size=None):
    """Load image and normalize to [-1, 1] as required by LPIPS"""
    img = Image.open(image_path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)

    # Convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to [0, 1]
        ]
    )
    tensor = transform(img)
    # Normalize to [-1, 1]
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)  # Add batch dimension


def compute_lpips_batch(
    real_images, generated_images, net="alex", device="cuda", batch_size=8
):
    """Compute LPIPS for pairs of images"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    lpips_values = []

    with torch.no_grad():
        for i in range(0, len(real_images), batch_size):
            batch_real = []
            batch_gen = []

            # Load batch of images
            for j in range(i, min(i + batch_size, len(real_images))):
                real_img = load_and_preprocess_image(real_images[j]).to(device)
                gen_img = load_and_preprocess_image(generated_images[j]).to(device)

                # Ensure same size (resize gen to match real if needed)
                if real_img.shape[-2:] != gen_img.shape[-2:]:
                    gen_img = F.interpolate(
                        gen_img,
                        size=real_img.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                batch_real.append(real_img)
                batch_gen.append(gen_img)

            # Stack into batch
            if batch_real:
                batch_real = torch.cat(batch_real, dim=0)
                batch_gen = torch.cat(batch_gen, dim=0)

                # Compute LPIPS for batch
                d = loss_fn(batch_real, batch_gen)
                lpips_values.extend(d.cpu().tolist())

    lpips_values = [float(v[0][0][0]) for v in lpips_values]
    return lpips_values


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_values = compute_lpips_batch(
        real_paths, gen_paths, net=args.net, device=device
    )
    avg_lpips = sum(lpips_values) / len(lpips_values)
    logger.info(f"LPIPS ({args.net}): {avg_lpips:.4f} (n={len(lpips_values)})")

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

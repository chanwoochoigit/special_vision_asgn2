#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a combined PixArt base (transformer + VAE/T5) in one local folder"
    )
    p.add_argument(
        "--transformer_repo",
        type=str,
        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        help="HF repo containing transformer subfolder",
    )
    p.add_argument(
        "--vae_t5_repo",
        type=str,
        default="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        help="HF repo containing vae, tokenizer, text_encoder, scheduler",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="PixArt-sigma/output/pretrained_models/pixart_sigma_combined",
        help="Local output directory to assemble combined base",
    )
    return p.parse_args()


def safe_copy_tree(src: Path, dst: Path):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = item
        d = dst / item.name
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tx_cache = Path(snapshot_download(args.transformer_repo))
    vt_cache = Path(snapshot_download(args.vae_t5_repo))

    safe_copy_tree(tx_cache / "transformer", out / "transformer")
    for sub in ["vae", "tokenizer", "text_encoder", "scheduler"]:
        safe_copy_tree(vt_cache / sub, out / sub)
    for filename in ["model_index.json", "README.md"]:
        src = vt_cache / filename
        if src.exists():
            shutil.copy2(src, out / filename)

    print(f"Combined base written to: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a combined PixArt base (transformer + VAE/T5) in one local folder"
    )
    p.add_argument(
        "--transformer_repo",
        type=str,
        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        help="HF repo containing transformer subfolder",
    )
    p.add_argument(
        "--vae_t5_repo",
        type=str,
        default="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        help="HF repo containing vae, tokenizer, text_encoder, scheduler",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="PixArt-sigma/output/pretrained_models/pixart_sigma_combined",
        help="Local output directory to assemble combined base",
    )
    return p.parse_args()


def safe_copy_tree(src: Path, dst: Path):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        s = item
        d = dst / item.name
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Download repos to cache
    tx_cache = Path(snapshot_download(args.transformer_repo))
    vt_cache = Path(snapshot_download(args.vae_t5_repo))

    # Copy transformer
    safe_copy_tree(tx_cache / "transformer", out / "transformer")

    # Copy VAE/T5/scheduler
    for sub in ["vae", "tokenizer", "text_encoder", "scheduler"]:
        safe_copy_tree(vt_cache / sub, out / sub)

    # Also copy pipeline-level config to satisfy DiffusionPipeline.from_pretrained at end of training
    for filename in ["model_index.json", "README.md"]:
        src = vt_cache / filename
        if src.exists():
            shutil.copy2(src, out / filename)

    print(f"Combined base written to: {out}")


if __name__ == "__main__":
    main()

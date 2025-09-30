#!/usr/bin/env python3
"""
Refactored, standalone benchmark script for SDXL vs. Infinity performance comparison.
Measures model size, inference time, and FID score on the Gundam dataset.
"""

import time
import os
import sys
import random
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline
from torchvision import transforms, models
from scipy.linalg import sqrtm

# --- Configuration ---
NUM_SAMPLES = 500  # Set to None to use the entire dataset
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_BASE_DIR = "local_repo"
INFINITY_REPO_DIR = os.path.join(os.path.dirname(__file__), "Infinity")
INCLUDE_INFINITY = os.path.isdir(
    INFINITY_REPO_DIR
)  # Automatically check if Infinity repo exists

# --- Utility Functions ---


def seed_everything(seed=0):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_gundam_dataset() -> Tuple[List[Image.Image], List[str], int]:
    """Loads and preprocesses the Gundam dataset from Hugging Face."""
    dataset = load_dataset("Gazoche/gundam-captioned", split="train")
    num_to_load = len(dataset) if NUM_SAMPLES is None else NUM_SAMPLES
    print(f"Using {num_to_load} samples from the dataset.")

    images = [
        img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        for img in dataset["image"][:num_to_load]
    ]
    captions = dataset["text"][:num_to_load]
    return images, captions, num_to_load


# --- FID Calculation Functions ---


def get_inception_features(images: List[Image.Image]) -> np.ndarray:
    """Extracts InceptionV3 features for a list of PIL images."""
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
    )
    inception.fc = nn.Identity()
    inception = inception.to(DEVICE).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features = []
    with torch.no_grad():
        for img in images:
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            feat = inception(tensor).cpu().numpy()
            features.append(feat)
    return np.vstack(features)


def calculate_fid(
    real_images: List[Image.Image], generated_images: List[Image.Image]
) -> float:
    """Calculates Fréchet Inception Distance between two sets of images."""
    print("Calculating FID score...")
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
    print(f"FID Calculation Complete. Score: {fid:.2f}")
    return fid


# --- Model Abstraction ---


class BenchmarkModel(ABC):
    """Abstract base class for a text-to-image model to be benchmarked."""

    def __init__(self):
        self.model_components = self.load()

    @abstractmethod
    def load(self) -> Any:
        """Loads the model and its components, returning them."""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> Image.Image:
        """Generates a single image from a prompt."""
        pass

    @abstractmethod
    def count_params(self) -> float:
        """Counts the trainable parameters of the core model in millions."""
        pass


class SDXLModel(BenchmarkModel):
    """Benchmark class for Stable Diffusion XL."""

    def load(self) -> Any:
        print("Loading SDXL model...")
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if DEVICE == "cuda" else None,
        )
        return pipe.to(DEVICE)

    def generate(self, prompt: str) -> Image.Image:
        result = self.model_components(
            prompt=prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=20,
            guidance_scale=7.5,
        )
        return result.images[0]

    def count_params(self) -> float:
        return sum(p.numel() for p in self.model_components.unet.parameters()) / 1e6


class InfinityModel(BenchmarkModel):
    """Benchmark class for FoundationVision/Infinity."""

    def load(self) -> Any:
        print("Loading Infinity model...")
        if not os.path.isdir(INFINITY_REPO_DIR):
            raise FileNotFoundError(
                "Infinity repo not found. Please clone it into the project root."
            )
        if INFINITY_REPO_DIR not in sys.path:
            sys.path.append(INFINITY_REPO_DIR)

        from types import SimpleNamespace

        try:
            from tools.run_infinity import (
                load_tokenizer as inf_load_tokenizer,
                load_visual_tokenizer as inf_load_vae,
                load_transformer as inf_load_transformer,
                gen_one_img,
            )
            from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
        except ImportError as e:
            raise ImportError(
                "Failed to import from Infinity repo. Ensure it is cloned correctly."
            ) from e

        weights_dir = os.path.join("local_repo", "Infinity")
        model_path = os.path.join(weights_dir, "infinity_2b_reg.pth")
        vae_path = os.path.join(weights_dir, "infinity_vae_d32reg.pth")

        if not os.path.exists(model_path) or not os.path.exists(vae_path):
            raise FileNotFoundError(f"Missing Infinity weights in {weights_dir}")

        args = SimpleNamespace(
            pn="0.25M",
            model_type="infinity_2b",
            vae_type=32,
            vae_path=vae_path,
            model_path=model_path,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            use_bit_label=1,
            add_lvl_embeding_only_first_block=0,
            text_channels=2048,
            apply_spatial_patchify=0,
            use_flex_attn=0,
            bf16=1,
            checkpoint_type="torch",
            cache_dir="/dev/shm",
            enable_model_cache=0,
            sampling_per_bits=1,
            text_encoder_ckpt="google/flan-t5-xl",
        )

        text_tokenizer, text_encoder = inf_load_tokenizer(
            t5_path=args.text_encoder_ckpt
        )
        vae = inf_load_vae(args)
        infinity_model = inf_load_transformer(vae, args)
        scale_schedule = [
            (1, h, w) for (_, h, w) in dynamic_resolution_h_w[1.0][args.pn]["scales"]
        ]

        return {
            "model": infinity_model,
            "vae": vae,
            "text_tokenizer": text_tokenizer,
            "text_encoder": text_encoder,
            "scale_schedule": scale_schedule,
            "gen_one_img": gen_one_img,
        }

    def generate(self, prompt: str) -> Image.Image:
        comps = self.model_components
        _n_scales = len(comps["scale_schedule"])

        img_tensor = comps["gen_one_img"](
            comps["model"],
            comps["vae"],
            comps["text_tokenizer"],
            comps["text_encoder"],
            prompt,
            cfg_list=[3.0] * _n_scales,
            tau_list=[1.0] * _n_scales,
            top_k=900,
            top_p=0.97,
            cfg_sc=3,
            cfg_insertion_layer=-5,
            vae_type=0,
            sampling_per_bits=1,
            scale_schedule=comps["scale_schedule"],
        )
        return Image.fromarray(img_tensor)

    def count_params(self) -> float:
        return sum(p.numel() for p in self.model_components["model"].parameters()) / 1e6


# --- Core Benchmark and Reporting Logic ---


def run_benchmark(
    model: BenchmarkModel,
    model_name: str,
    images: List[Image.Image],
    captions: List[str],
) -> Dict[str, Any]:
    """Runs the benchmark for a given model, returning performance metrics."""
    generated_images = []
    inference_times = []
    output_dir = os.path.join(OUTPUT_BASE_DIR, model_name, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*20} Benchmarking {model_name} {'='*20}")
    print(f"Saving generated images to: {output_dir}")
    overall_start_time = time.time()

    with torch.no_grad():
        for i, caption in enumerate(captions):
            iter_start = time.time()
            gen_img = model.generate(prompt=caption)
            iter_end = time.time()

            inference_times.append(iter_end - iter_start)
            generated_images.append(gen_img)
            gen_img.save(os.path.join(output_dir, f"{i:05d}.png"))

            if (i + 1) % 100 == 0:
                avg_time = np.mean(inference_times)
                print(
                    f"  Processed {i + 1}/{len(captions)} samples (avg: {avg_time:.3f}s/image)"
                )

    overall_end_time = time.time()
    total_inference_time = overall_end_time - overall_start_time

    # --- Calculate metrics after the loop ---
    fid_score = calculate_fid(images, generated_images)
    avg_time = np.mean(inference_times)

    results = {
        "model_size": model.count_params(),
        "avg_inference_time": avg_time,
        "std_inference_time": np.std(inference_times),
        "min_inference_time": np.min(inference_times),
        "max_inference_time": np.max(inference_times),
        "total_inference_time": total_inference_time,
        "throughput": len(captions) / total_inference_time,
        "fid_score": fid_score,
    }

    print(
        f"✅ {model_name} benchmark complete. Total time: {total_inference_time:.2f}s"
    )
    return results


def print_results(
    sdxl_results: Dict[str, float],
    other_results: Dict[str, float],
    other_model_name: str,
):
    """Prints a formatted comparison table of the benchmark results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<35} {'SDXL':<20} {other_model_name:<20}")
    print("-" * 80)
    for key, label in [
        ("model_size", "Model Size (M params)"),
        ("avg_inference_time", "Avg Inference Time (s)"),
        ("std_inference_time", "Std Inference Time (s)"),
        ("throughput", "Throughput (images/sec)"),
        ("total_inference_time", "Total Inference Time (s)"),
        ("fid_score", "FID Score (lower = better)"),
    ]:
        print(
            f"{label:<35} {sdxl_results.get(key, 0):<20.2f} {other_results.get(key, 0):<20.2f}"
        )
    print("=" * 80)


def save_comprehensive_report(results: Dict[str, Dict], num_samples: int):
    """Saves a comprehensive JSON and TXT report of the final results."""
    report_dir = os.path.join(OUTPUT_BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_data = {
        "timestamp": timestamp,
        "configuration": {
            "num_samples": num_samples,
            "image_size": IMAGE_SIZE,
            "device": DEVICE,
        },
        "results": results,
    }

    # Save JSON report
    json_path = os.path.join(report_dir, f"benchmark_report_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n📊 Comprehensive JSON report saved to: {json_path}")


# --- Main Execution ---


def main():
    """Main execution function with memory-efficient sequential processing."""
    seed_everything(42)
    print(f"Using device: {DEVICE}")
    print("=" * 80)

    images, captions, num_samples = load_gundam_dataset()
    all_results = {}

    # --- Benchmark SDXL ---
    try:
        sdxl_model = SDXLModel()
        sdxl_results = run_benchmark(sdxl_model, "SDXL", images, captions)
        all_results["SDXL"] = sdxl_results
        del sdxl_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("🧹 Cleared SDXL from memory")
    except Exception as e:
        print(f"❌ Failed to benchmark SDXL: {e}")
        sdxl_results = None

    # --- Benchmark Infinity ---
    if INCLUDE_INFINITY:
        try:
            infinity_model = InfinityModel()
            infinity_results = run_benchmark(
                infinity_model, "Infinity", images, captions
            )
            all_results["Infinity"] = infinity_results
            del infinity_model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print("🧹 Cleared Infinity from memory")
        except Exception as e:
            print(f"❌ Failed to benchmark Infinity: {e}")
            infinity_results = None
    else:
        print("\n⚠️ Infinity repo not found, skipping Infinity benchmark.")
        infinity_results = None

    # --- Final Reporting ---
    if sdxl_results and infinity_results:
        print_results(sdxl_results, infinity_results, "Infinity")
    elif sdxl_results:
        print("\nOnly SDXL results are available.")

    if all_results:
        save_comprehensive_report(all_results, num_samples)

    print("\n🎉 Benchmark completed successfully!")


if __name__ == "__main__":
    main()

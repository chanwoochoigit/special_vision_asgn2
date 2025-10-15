#!/usr/bin/env python3

import time
import os
import sys
import random
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
import warnings

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline, Transformer2DModel, PixArtSigmaPipeline
from torchvision import transforms, models
from scipy.linalg import sqrtm

# --- Configuration ---
NUM_SAMPLES = None
IMAGE_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_BASE_DIR = "local_repo"
PIXART_REPO_DIR = os.path.join(os.path.dirname(__file__), "PixArt-sigma")
INCLUDE_PIXART = os.path.isdir(PIXART_REPO_DIR)

# Runtime toggles (default: run only PixArt)
SKIP_SDXL = os.getenv("SKIP_SDXL", "1") == "1"
RUN_PIXART = os.getenv("RUN_PIXART", "1") == "1"

# Optional override for number of samples via environment
_env_num = os.getenv("NUM_SAMPLES")
if _env_num is not None and _env_num.strip() != "":
    NUM_SAMPLES = int(_env_num)


# --- Utility Functions ---
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_gundam_dataset() -> Tuple[List[Image.Image], List[str], int]:
    dataset = load_dataset("Gazoche/gundam-captioned", split="train")
    num_to_load = len(dataset) if NUM_SAMPLES is None else NUM_SAMPLES
    print(f"Using {num_to_load} samples from the dataset.")
    images = [
        img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        for img in dataset["image"][:num_to_load]
    ]
    captions = dataset["text"][:num_to_load]
    return images, captions, num_to_load


# --- FID Calculation ---
def get_inception_features(images: List[Image.Image]) -> np.ndarray:
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


# --- PixArt-Sigma Model Class ---
class PixArtSigmaModel:
    def __init__(self, model_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"):
        self.model_id = model_id
        self.pipe = self.load()

    def load(self) -> Any:
        print("Loading PixArt-Sigma model…")
        # Transformer weights
        transformer = Transformer2DModel.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            subfolder="transformer",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            use_safetensors=True,
        )
        # Base VAE + T5 package
        vae_t5_repo = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
        pipe = PixArtSigmaPipeline.from_pretrained(
            vae_t5_repo,
            transformer=transformer,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            use_safetensors=True,
        )
        return pipe.to(DEVICE)

    def generate(self, prompt: str) -> Image.Image:
        result = self.pipe(prompt, height=IMAGE_SIZE, width=IMAGE_SIZE)
        return result.images[0]

    def count_params(self) -> float:
        return sum(p.numel() for p in self.pipe.transformer.parameters()) / 1e6


# --- SDXL Model Class (Unchanged) ---
class SDXLModel:
    def __init__(self):
        self.model_components = self.load()

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


# --- Reporting and Result Handling ---
def print_results(sdxl_results, other_results, other_model_name):
    print("\n" + "=" * 80 + "\nCOMPREHENSIVE BENCHMARK RESULTS\n" + "=" * 80)
    print(f"{'Metric':<35} {'SDXL':<20} {other_model_name:<20}\n" + "-" * 80)
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


def save_comprehensive_report(results, num_samples):
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
    json_path = os.path.join(report_dir, f"benchmark_report_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n📊 Comprehensive JSON report saved to: {json_path}")


def save_model_results(model_name, results, num_samples):
    results_dir = os.path.join(OUTPUT_BASE_DIR, "intermediate_results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{model_name}_results.json")
    data_to_save = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "configuration": {
            "num_samples": num_samples,
            "image_size": IMAGE_SIZE,
            "device": DEVICE,
        },
        "results": results,
    }
    with open(file_path, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"✅ Saved intermediate results for {model_name} to {file_path}")


def load_model_results(model_name: str) -> Dict[str, Any]:
    results_dir = os.path.join(OUTPUT_BASE_DIR, "intermediate_results")
    file_path = os.path.join(results_dir, f"{model_name}_results.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded intermediate results for {model_name} from {file_path}")
    return data["results"]


# --- Main Execution ---
def main():
    # Suppress FutureWarning messages from libraries to clean up the output
    warnings.filterwarnings("ignore", category=FutureWarning)

    seed_everything(42)
    print(f"Using device: {DEVICE}\n" + "=" * 80)
    images, captions, num_samples = load_gundam_dataset()
    all_results = {}

    results_dir = os.path.join(OUTPUT_BASE_DIR, "intermediate_results")
    sdxl_results_path = os.path.join(results_dir, "SDXL_results.json")
    pixart_results_path = os.path.join(results_dir, "PixArt_results.json")

    # --- Benchmark SDXL (optional) ---
    sdxl_results = None
    if not SKIP_SDXL:
        if os.path.exists(sdxl_results_path):
            sdxl_results = load_model_results("SDXL")
        else:
            try:
                sdxl_model = SDXLModel()
                sdxl_results = run_benchmark_sdxl(sdxl_model, images, captions)
                save_model_results("SDXL", sdxl_results, num_samples)
                del sdxl_model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print("🧹 Cleared SDXL from memory")
            except Exception as e:
                print(f"❌ Failed to benchmark SDXL: {e}")
                sdxl_results = None
        if sdxl_results:
            all_results["SDXL"] = sdxl_results

    # --- Benchmark PixArt-Sigma (requested) ---
    pixart_results = None
    if RUN_PIXART and INCLUDE_PIXART:
        if os.path.exists(pixart_results_path):
            pixart_results = load_model_results("PixArt")
        else:
            try:
                pixart_model = PixArtSigmaModel()
                pixart_results = run_benchmark_pixart(pixart_model, images, captions)
                save_model_results("PixArt", pixart_results, num_samples)
                del pixart_model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print("🧹 Cleared PixArt-Sigma from memory")
            except Exception as e:
                print(f"❌ Failed to benchmark PixArt-Sigma: {e}")
                pixart_results = None
        if pixart_results:
            all_results["PixArt"] = pixart_results
    else:
        if not INCLUDE_PIXART:
            print("\n⚠️ PixArt-sigma repo not found, skipping PixArt benchmark.")

    # --- Final Reporting ---
    if "PixArt" in all_results:
        # If SDXL present, print comparison, else just print PixArt summary via report saving
        pass

    if all_results:
        save_comprehensive_report(all_results, num_samples)
    print("\n🎉 Benchmark completed successfully!")


# Helper wrapper for SDXL
def run_benchmark_sdxl(
    model: SDXLModel, images: List[Image.Image], captions: List[str]
) -> Dict[str, Any]:
    generated_images = []
    inference_times = []
    output_dir = os.path.join(OUTPUT_BASE_DIR, "SDXL", "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*20} Benchmarking SDXL {'='*20}")
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
                print(
                    f"  Processed {i + 1}/{len(captions)} samples (avg: {np.mean(inference_times):.3f}s/image)"
                )

    total_inference_time = time.time() - overall_start_time
    fid_score = calculate_fid(images, generated_images)

    results = {
        "model_size": model.count_params(),
        "avg_inference_time": np.mean(inference_times),
        "std_inference_time": np.std(inference_times),
        "min_inference_time": np.min(inference_times),
        "max_inference_time": np.max(inference_times),
        "total_inference_time": total_inference_time,
        "throughput": len(captions) / total_inference_time,
        "fid_score": fid_score,
    }
    print(f"✅ SDXL benchmark complete. Total time: {total_inference_time:.2f}s")
    return results


# Helper wrapper for PixArt-Sigma
def run_benchmark_pixart(
    model: PixArtSigmaModel, images: List[Image.Image], captions: List[str]
) -> Dict[str, Any]:
    generated_images = []
    inference_times = []
    output_dir = os.path.join(OUTPUT_BASE_DIR, "PixArt", "output")
    input_dir = os.path.join(OUTPUT_BASE_DIR, "PixArt", "input")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    print(f"\n{'='*20} Benchmarking PixArt-Sigma {'='*20}")
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

            # Save corresponding input image and caption for traceability
            images[i].save(os.path.join(input_dir, f"{i:05d}_input.png"))

            with open(os.path.join(input_dir, f"{i:05d}_caption.txt"), "w") as cf:
                cf.write(str(caption))
            if (i + 1) % 100 == 0:
                print(
                    f"  Processed {i + 1}/{len(captions)} samples (avg: {np.mean(inference_times):.3f}s/image)"
                )

    total_inference_time = time.time() - overall_start_time
    fid_score = calculate_fid(images, generated_images)

    results = {
        "model_size": model.count_params(),
        "avg_inference_time": np.mean(inference_times),
        "std_inference_time": np.std(inference_times),
        "min_inference_time": np.min(inference_times),
        "max_inference_time": np.max(inference_times),
        "total_inference_time": total_inference_time,
        "throughput": len(captions) / total_inference_time,
        "fid_score": fid_score,
    }
    print(
        f"✅ PixArt-Sigma benchmark complete. Total time: {total_inference_time:.2f}s"
    )
    return results


if __name__ == "__main__":
    main()

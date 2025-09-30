# SDXL vs VAR Model Performance Benchmark

A minimalist, clean Python script to benchmark Stable Diffusion XL (SDXL) and the **NeurIPS 2024 Best Paper Award** winning Visual Autoregressive (VAR) model on the Gundam dataset.

## Features

- **Model Comparison**: Benchmarks SDXL vs FoundationVision/VAR (NeurIPS 2024)
- **Comprehensive Metrics**: Model size, inference time, and FID score with detailed analysis
- **Minimal & Clean**: Follows separation of concerns and modular design
- **GPU Optimized**: Automatic GPU detection with memory management

## Installation

```bash
pip install -r requirements.txt
```

**Note**: The VAR model must be properly installed. The script will fail if VAR dependencies are not available.

## Usage

```bash
python stuff.py
```

## Output

The script generates comprehensive benchmark results including:

### Core Metrics:
- **Model Size**: Total parameters in millions
- **Average Inference Time**: Mean time per image in seconds
- **Standard Deviation**: Inference time variability
- **Min/Max Inference Time**: Performance range
- **Throughput**: Images generated per second
- **FID Score**: Image quality metric (lower = better)

### Performance Analysis:
- **Speed Winner**: Fastest model with speed ratio
- **Quality Winner**: Best image quality with FID difference
- **Efficiency Winner**: Best throughput per parameter ratio

### Sample Output:
```
COMPREHENSIVE BENCHMARK RESULTS
================================================================================
Metric                              SDXL                 VAR-d16             
--------------------------------------------------------------------------------
Model Size (M params)               2600.0               310.0               
Avg Inference Time (s)              3.45                 1.89                
Std Inference Time (s)              0.23                 0.15                
Min Inference Time (s)              2.89                 1.45                
Max Inference Time (s)              4.12                 2.34                
Throughput (images/sec)             0.29                 0.53                
FID Score (lower = better)          15.2                 12.8                

PERFORMANCE ANALYSIS:
----------------------------------------
Speed Winner: VAR-d16 (1.8x faster)
Quality Winner: VAR-d16 (FID lower by 2.4)
Efficiency Winner: VAR-d16 (3.7x more efficient)
```


## Configuration

Edit the configuration variables at the top of `stuff.py`:
- `NUM_SAMPLES`: Number of test samples (default: 50)
- `IMAGE_SIZE`: Generated image resolution (default: 512)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ~8GB GPU memory for optimal performance

## Models Used

- **SDXL**: `stabilityai/stable-diffusion-xl-base-1.0` - State-of-the-art diffusion model
- **VAR**: [FoundationVision/VAR](https://github.com/FoundationVision/VAR) - NeurIPS 2024 Best Paper Award winner
  - Uses VAR-d16 configuration (310M parameters) for balanced speed and quality
  - Implements next-scale autoregressive prediction for image generation

## Citation

If you use this benchmarking script in your research, please cite the VAR paper:

```bibtex
@Article{VAR,
      title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
      author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
      year={2024},
      eprint={2404.02905},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
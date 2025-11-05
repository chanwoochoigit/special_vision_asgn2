#!/usr/bin/env bash
PYTHON_BIN=${PYTHON_BIN:-python}
RAW_DIR=${RAW_DIR:-local_repo/WikiArt/output}
GEN_DIR=${GEN_DIR:-local_repo/WikiArt_ft/output}
REAL_DIR=${REAL_DIR:-local_repo/WikiArt/input/test}
LOG_FILE=${LOG_FILE:-logs/metrics_wikiart.log}

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Suppress Python warnings in log file
export PYTHONWARNINGS=ignore

# Redirect all output to log file (and still show on screen with tee)
# Use a subshell approach that works with both sh and bash
(
echo "=== Metrics computation started at $(date) ==="

echo "Computing FID for raw model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_fid_for_folder.py --generated_dir "$RAW_DIR" --real_dir "$REAL_DIR" --print_timing_stats
echo "Computing FID for finetuned model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_fid_for_folder.py --generated_dir "$GEN_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing FD-DINOv2 for raw model..."
(cd FD-DINOv2 && PYTHONPATH=".:src/pytorch_fd" "$PYTHON_BIN" src/pytorch_fd/fd_score.py "../$RAW_DIR" "../$REAL_DIR")

echo "Computing FD-DINOv2 for finetuned model..."
(cd FD-DINOv2 && PYTHONPATH=".:src/pytorch_fd" "$PYTHON_BIN" src/pytorch_fd/fd_score.py "../$GEN_DIR" "../$REAL_DIR")

echo "Computing LPIPS for raw model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_lpips_for_folder.py --generated_dir "$RAW_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing LPIPS for finetuned model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_lpips_for_folder.py --generated_dir "$GEN_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing SSIM for raw model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_ssim_for_folder.py --generated_dir "$RAW_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing SSIM for finetuned model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_ssim_for_folder.py --generated_dir "$GEN_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing PSNR for raw model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_psnr_for_folder.py --generated_dir "$RAW_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing PSNR for finetuned model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_psnr_for_folder.py --generated_dir "$GEN_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing Delta E 2000 for raw model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_deltae2000_for_folder.py --generated_dir "$RAW_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "Computing Delta E 2000 for finetuned model..."
"$PYTHON_BIN" scripts/pixart_scripts/common/metrics/compute_deltae2000_for_folder.py --generated_dir "$GEN_DIR" --real_dir "$REAL_DIR" --print_timing_stats

echo "=== Metrics computation completed at $(date) ==="
echo "Results saved to: $LOG_FILE"
) 2>&1 | tee -a "$LOG_FILE"

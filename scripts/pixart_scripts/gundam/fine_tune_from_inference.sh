#!/usr/bin/env bash
set -euo pipefail

# One-touch pipeline: assumes run_inference.py already created local_repo/PixArt/output via RUN_PIXART=1
# Steps:
# 1) Backfill inputs (images+captions) matching outputs
# 2) Create imagefolder metadata.jsonl
# 3) Clean conflicting global/user packages
# 4) Install training deps using active python
# 5) Build combined PixArt base (transformer + VAE/T5)
# 6) Launch LoRA training directly from local_repo/PixArt/input

export PYTHONNOUSERSITE=1
PYTHON_BIN=${PYTHON_BIN:-python}

BASE_PIXART=${BASE_PIXART:-local_repo/PixArt}
INPUT_DIR="$BASE_PIXART/input"
OUTPUT_DIR=${OUTPUT_DIR:-PixArt-sigma/output/gundam_lora_512}
COMBINED_BASE=${COMBINED_BASE:-PixArt-sigma/output/pretrained_models/pixart_sigma_combined}
PORT=${PORT:-12345}
BS=${BS:-2}
STEPS=${STEPS:-2000}
LR=${LR:-1e-4}
RANK=${RANK:-8}
VALIDATION_PROMPT=${VALIDATION_PROMPT:-"A realistic Gundam mecha standing in a city, highly detailed"}

echo "[1/6] Backfilling inputs from Gundam dataset to match generated outputs..."
"$PYTHON_BIN" scripts/pixart_scripts/gundam/backfill_pixart_inputs.py --base_dir "$BASE_PIXART" --dataset Gazoche/gundam-captioned --split train --image_size 512

echo "[2/6] Generating metadata.jsonl in $INPUT_DIR ..."
"$PYTHON_BIN" scripts/pixart_scripts/gundam/make_imagefolder_metadata.py --input_dir "$INPUT_DIR"

echo "[3/6] Cleaning conflicting packages (xformers/flash-attn) ..."
bash scripts/pixart_scripts/gundam/clean_conflicting_pkgs.sh

echo "[4/6] Installing training dependencies ..."
bash scripts/pixart_scripts/gundam/install_training_deps.sh

echo "[4.5/6] Verifying training environment ..."
"$PYTHON_BIN" - << 'PY'
import importlib
mods = ["torch","diffusers","transformers","accelerate","datasets","peft"]
ok=True
for m in mods:
    try:
        mod=importlib.import_module(m)
        v=getattr(mod,'__version__','?')
        print(f"OK {m}: {v}")
    except Exception as e:
        ok=False
        print(f"FAIL {m}: {e}")
raise SystemExit(0 if ok else 1)
PY

echo "[5/6] Building combined PixArt base at $COMBINED_BASE ..."
"$PYTHON_BIN" scripts/pixart_scripts/gundam/build_pixart_combined_base.py --out_dir "$COMBINED_BASE"

echo "[6/6] Launching LoRA training from $INPUT_DIR ..."
PYTHON_BIN="$PYTHON_BIN" BASE_DIR="$INPUT_DIR" OUT_DIR="$OUTPUT_DIR" BASE_COMBINED="$COMBINED_BASE" PORT="$PORT" BS="$BS" STEPS="$STEPS" LR="$LR" RANK="$RANK" \
  VALIDATION_PROMPT="$VALIDATION_PROMPT" bash scripts/pixart_scripts/gundam/train_pixart_lora_from_inputs.sh

echo "[7/7] Validating LoRA and generating new inferences + FID ..."
NEW_OUT_DIR=${NEW_OUT_DIR:-local_repo/PixArt_ft/output_new}
"$PYTHON_BIN" scripts/pixart_scripts/gundam/validate_finetune.py --base_combined "$COMBINED_BASE" --lora_dir "$OUTPUT_DIR" --out_dir "${NEW_OUT_DIR}/validation"
"$PYTHON_BIN" scripts/pixart_scripts/gundam/infer_finetuned_on_new_gundam.py --base_combined "$COMBINED_BASE" --lora_dir "$OUTPUT_DIR" --out_dir "$NEW_OUT_DIR" --num_samples 300 --skip_first_n 1028
"$PYTHON_BIN" scripts/pixart_scripts/gundam/compute_fid_for_folder.py --generated_dir "$NEW_OUT_DIR" --start 1028 --count 300

echo "Done. LoRA weights at: $OUTPUT_DIR"



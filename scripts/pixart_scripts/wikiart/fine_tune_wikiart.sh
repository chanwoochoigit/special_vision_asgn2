#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
PYTHON_BIN=${PYTHON_BIN:-python}

BASE_WIKIART=${BASE_WIKIART:-local_repo/WikiArt}
# Train on the prepared train split
INPUT_DIR="$BASE_WIKIART/input/train"
RAW_INPUT_DIR="$BASE_WIKIART/input/test"
RAW_OUT_DIR="$BASE_WIKIART/output"
OUTPUT_DIR=${OUTPUT_DIR:-PixArt-sigma/output/wikiart_lora_512}
COMBINED_BASE=${COMBINED_BASE:-PixArt-sigma/output/pretrained_models/pixart_sigma_combined}
PORT=${PORT:-12345}
BS=${BS:-2}
STEPS=${STEPS:-3000}
LR=${LR:-5e-5}3
RANK=${RANK:-32}
VALIDATION_PROMPT=${VALIDATION_PROMPT:-"a painting of a city street in impressionist style"}

echo "[1/6] Preparing WikiArt inputs ..."
"$PYTHON_BIN" scripts/pixart_scripts/wikiart/prepare_inputs_from_dataset.py --base_dir "$BASE_WIKIART" --dataset huggan/wikiart --split train --num_samples 2000 --image_size 512 --caption_field style

echo "[2/6] Generating metadata.jsonl in $INPUT_DIR ..."
"$PYTHON_BIN" scripts/pixart_scripts/common/make_imagefolder_metadata.py --input_dir "$INPUT_DIR"

echo "[3/6] Cleaning conflicting packages ..."
bash scripts/pixart_scripts/common/clean_conflicting_pkgs.sh

echo "[4/6] Installing training deps ..."
bash scripts/pixart_scripts/common/install_training_deps.sh

echo "[5/6] Building combined base at $COMBINED_BASE ..."
"$PYTHON_BIN" scripts/pixart_scripts/common/build_pixart_combined_base.py --out_dir "$COMBINED_BASE"

echo "[6/6] Launching LoRA training from $INPUT_DIR ..."
PYTHON_BIN="$PYTHON_BIN" BASE_DIR="$INPUT_DIR" OUT_DIR="$OUTPUT_DIR" BASE_COMBINED="$COMBINED_BASE" PORT="$PORT" BS="$BS" STEPS="$STEPS" LR="$LR" RANK="$RANK" \
  VALIDATION_PROMPT="$VALIDATION_PROMPT" bash scripts/pixart_scripts/wikiart/train_pixart_lora_from_inputs.sh
echo "Done. LoRA weights at: $OUTPUT_DIR"

echo "[6.5/6] Inferring with raw model ..."
"$PYTHON_BIN" scripts/pixart_scripts/wikiart/infer_raw_model.py --input_dir "$RAW_INPUT_DIR" --out_dir "$RAW_OUT_DIR" --cfg 4.0

echo "[7/7] Validating LoRA and generating new inferences + FID ..."
NEW_OUT_DIR=${NEW_OUT_DIR:-local_repo/WikiArt_ft/output}
"$PYTHON_BIN" scripts/pixart_scripts/wikiart/validate_finetune.py --base_combined "$COMBINED_BASE" --lora_dir "$OUTPUT_DIR" --out_dir "${NEW_OUT_DIR}/validation"
"$PYTHON_BIN" scripts/pixart_scripts/wikiart/infer_finetuned_on_wikiart.py --base_combined "$COMBINED_BASE" --lora_dir "$OUTPUT_DIR" --out_dir "$NEW_OUT_DIR" --cfg 4.0
bash scripts/pixart_scripts/wikiart/compute_fid_wikiart.sh



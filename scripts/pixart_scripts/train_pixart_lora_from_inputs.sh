#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON_BIN=${PYTHON_BIN:-python3}

BASE_DIR=${BASE_DIR:-local_repo/PixArt/input}
OUT_DIR=${OUT_DIR:-PixArt-sigma/output/gundam_lora_512}
BASE_COMBINED=${BASE_COMBINED:-PixArt-sigma/output/pretrained_models/pixart_sigma_combined}
PORT=${PORT:-12345}
BS=${BS:-2}
STEPS=${STEPS:-2000}
LR=${LR:-1e-4}
RANK=${RANK:-8}
VALIDATION_PROMPT=${VALIDATION_PROMPT:-"A realistic Gundam mecha standing in a city, highly detailed"}

"$PYTHON_BIN" scripts/pixart_scripts/make_imagefolder_metadata.py --input_dir "$BASE_DIR"
"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" \
  PixArt-sigma/train_scripts/train_pixart_lora_hf.py \
  --pretrained_model_name_or_path "$BASE_COMBINED" \
  --dataset_name imagefolder \
  --train_data_dir "$BASE_DIR" \
  --resolution 512 --micro_conditions --mixed_precision fp16 \
  --train_batch_size "$BS" --gradient_checkpointing \
  --max_train_steps "$STEPS" --learning_rate "$LR" --lr_scheduler cosine --lr_warmup_steps 100 \
  --rank "$RANK" \
  --validation_prompt "$VALIDATION_PROMPT" --num_validation_images 1 \
  --output_dir "$OUT_DIR"

echo "LoRA training completed. Output at: $OUT_DIR"

#!/usr/bin/env bash
set -euo pipefail

# Ensure we don't accidentally import user-level packages (e.g., xformers from ~/.local)
export PYTHONNOUSERSITE=1

# Use the current environment's Python by default
PYTHON_BIN=${PYTHON_BIN:-python3}

# Training wrapper that reads directly from local_repo/PixArt/input

BASE_DIR=${BASE_DIR:-local_repo/PixArt/input}
OUT_DIR=${OUT_DIR:-PixArt-sigma/output/gundam_lora_512}
BASE_COMBINED=${BASE_COMBINED:-PixArt-sigma/output/pretrained_models/pixart_sigma_combined}
PORT=${PORT:-12345}
BS=${BS:-2}
STEPS=${STEPS:-2000}
LR=${LR:-1e-4}
RANK=${RANK:-8}
VALIDATION_PROMPT=${VALIDATION_PROMPT:-"A realistic Gundam mecha standing in a city, highly detailed"}

# Ensure metadata exists for imagefolder

"$PYTHON_BIN" scripts/make_imagefolder_metadata.py --input_dir "$BASE_DIR"
"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" \
  PixArt-sigma/train_scripts/train_pixart_lora_hf.py \
  --pretrained_model_name_or_path "$BASE_COMBINED" \
  --dataset_name imagefolder \
  --train_data_dir "$BASE_DIR" \
  --resolution 512 --micro_conditions --mixed_precision fp16 \
  --train_batch_size "$BS" --gradient_checkpointing \
  --max_train_steps "$STEPS" --learning_rate "$LR" --lr_scheduler cosine --lr_warmup_steps 100 \
  --rank "$RANK" \
  --validation_prompt "$VALIDATION_PROMPT" --num_validation_images 1 \
  --output_dir "$OUT_DIR"

echo "LoRA training completed. Output at: $OUT_DIR"



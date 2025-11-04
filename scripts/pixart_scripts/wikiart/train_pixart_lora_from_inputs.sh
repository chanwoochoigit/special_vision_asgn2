#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON_BIN=${PYTHON_BIN:-python}

BASE_DIR=${BASE_DIR:-local_repo/WikiArt/input/train}
OUT_DIR=${OUT_DIR:-PixArt-sigma/output/wikiart_lora_512}
BASE_COMBINED=${BASE_COMBINED:-PixArt-sigma/output/pretrained_models/pixart_sigma_combined}
PORT=${PORT:-12345}
BS=${BS:-2}
STEPS=${STEPS:-3000}
LR=${LR:-5e-5}
RANK=${RANK:-32}
VALIDATION_PROMPT=${VALIDATION_PROMPT:-"Artist: berthe-morisot, Style: Impressionism, Genre: landscape"}

"$PYTHON_BIN" scripts/pixart_scripts/common/make_imagefolder_metadata.py --input_dir "$BASE_DIR"
"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" \
  PixArt-sigma/train_scripts/train_pixart_lora_hf.py \
  --pretrained_model_name_or_path "$BASE_COMBINED" \
  --dataset_name imagefolder \
  --train_data_dir "$BASE_DIR" \
  --resolution 512 --micro_conditions --mixed_precision fp16 --random_flip \
  --train_batch_size "$BS" --gradient_checkpointing \
  --max_train_steps "$STEPS" --learning_rate "$LR" --lr_scheduler cosine --lr_warmup_steps 100 \
  --rank "$RANK" --use_rslora --use_dora \
  --validation_prompt "$VALIDATION_PROMPT" --num_validation_images 1 \
  --output_dir "$OUT_DIR"

echo "LoRA training completed. Output at: $OUT_DIR"
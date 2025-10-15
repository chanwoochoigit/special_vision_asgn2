#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" -m pip install -U pip wheel setuptools
"$PYTHON_BIN" -m pip install -U torch torchvision
"$PYTHON_BIN" -m pip install -U diffusers==0.30.0 transformers>=4.41.0 accelerate>=0.33.0 sentencepiece safetensors huggingface_hub datasets peft
"$PYTHON_BIN" -m pip uninstall -y xformers flash-attn || true
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "Training dependencies installed."

#!/usr/bin/env bash
set -euo pipefail

# Use the active environment's python for pip installs
PYTHON_BIN=${PYTHON_BIN:-python3}

# Pin compatible versions for diffusers training (LoRA) on 4090

"$PYTHON_BIN" -m pip install -U pip wheel setuptools

# Core libs
"$PYTHON_BIN" -m pip install -U torch torchvision

# Hugging Face + training stack
"$PYTHON_BIN" -m pip install -U diffusers==0.30.0 transformers>=4.41.0 accelerate>=0.33.0 sentencepiece safetensors huggingface_hub datasets

# Ensure we do NOT pull optional CUDA-heavy extras that break on some setups
"$PYTHON_BIN" -m pip uninstall -y xformers flash-attn || true

# Install project requirements (includes peft)
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "Training dependencies installed."



#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" -m pip install -U pip wheel setuptools
"$PYTHON_BIN" -m pip install -U torch torchvision
"$PYTHON_BIN" -m pip install -U diffusers==0.30.0 transformers>=4.41.0 accelerate>=0.33.0 sentencepiece safetensors huggingface_hub datasets peft
"$PYTHON_BIN" -m pip uninstall -y xformers flash-attn || true
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "Training dependencies installed."



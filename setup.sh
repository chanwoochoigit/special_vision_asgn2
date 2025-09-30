#!/usr/bin/env bash
set -e

# This script automates the setup process for the benchmark environment.
# It clones the required repository, creates directories, downloads model weights,
# and installs Python dependencies.

# 1. Clone the Infinity repository
echo " Cloning Infinity repository..."
git clone https://github.com/FoundationVision/Infinity.git

# 2. Create the directory for the weights
# The script expects the weights to be in 'local_repo_refactored/Infinity/'.
echo " Creating directory for model weights..."
mkdir -p local_repo_refactored/Infinity

# 3. Download the Infinity model and VAE weights
# We use wget to download the files directly into the correct folder.
echo " Downloading Infinity 2B model weights (3.8 GB)..."
wget -c -O local_repo_refactored/Infinity/infinity_2b_reg.pth https://huggingface.co/FoundationVision/Infinity-2B/resolve/main/infinity_2b_reg.pth

echo " Downloading Infinity VAE weights (158 MB)..."
wget -c -O local_repo_refactored/Infinity/infinity_vae_d32reg.pth https://huggingface.co/FoundationVision/Infinity-2B/resolve/main/infinity_vae_d32reg.pth

# 4. Install all Python dependencies from the requirements file
echo " Installing Python packages..."
pip install -r requirements.txt

echo "Setup complete! You can now run the benchmark script. 🎉"
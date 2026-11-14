#!/usr/bin/env bash
# ==============================================================================
# setup_wsl.sh — Reproducible WSL2 Ubuntu 24.04 Environment Setup for TRIBE-local
# Target GPU: NVIDIA GeForce RTX 5060 Ti (8GB VRAM)
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Setting up TRIBE-local on WSL2 Ubuntu 24.04"
echo "=========================================================="

# 1. Update system packages and install system prerequisites
echo "[1/6] Updating system packages..."
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    ca-certificates

# 2. Check NVIDIA CUDA driver pass-through in WSL2
echo "[2/6] Checking NVIDIA GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not detected in WSL2. Please ensure NVIDIA Windows drivers are installed."
fi

# 3. Install Miniconda if conda is not found
echo "[3/6] Configuring Conda environment..."
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    CONDA_DIR="$HOME/miniconda3"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    export PATH="$CONDA_DIR/bin:$PATH"
    eval "$($CONDA_DIR/bin/conda shell.bash hook)"
fi

# 4. Create conda environment
ENV_NAME="tribe_local"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Activating..."
else
    echo "Creating Conda environment '$ENV_NAME' with Python 3.11 and PyTorch CUDA 12.1..."
    conda create -y -n "$ENV_NAME" python=3.11
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 5. Install PyTorch with CUDA 12.1
echo "[4/6] Installing PyTorch with CUDA 12.1..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Install Project Dependencies
echo "[5/6] Installing TRIBE-local Python requirements..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

pip install -r requirements.txt
pip install -e .

# 7. Create Cache and Data Directories
echo "[6/6] Creating cache directories..."
mkdir -p "$HOME/.tribe_cache/video"
mkdir -p "$HOME/.tribe_cache/audio"
mkdir -p "$HOME/.tribe_cache/text"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/checkpoints"
mkdir -p "$PROJECT_ROOT/logs"

echo "=========================================================="
echo " Setup Complete!"
echo " To activate: conda activate tribe_local"
echo " Run benchmark: python -m tribe_local.evaluation.benchmark"
echo "=========================================================="

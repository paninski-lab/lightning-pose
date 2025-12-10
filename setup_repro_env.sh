#!/bin/bash
set -e  # Exit immediately if a command fails

# --- Configuration ---
ENV_NAME="lp_repro"
PYTHON_VER="3.10"

echo ">>> Loading Conda/Mamba module..."
module load Mambaforge/23.11.0-fasrc01

echo ">>> Creating environment: $ENV_NAME..."
mamba create -n "$ENV_NAME" python="$PYTHON_VER" -y

echo ">>> Activating environment..."
source activate "$ENV_NAME"

# --- CRITICAL SAFETY CHECKS ---
export PYTHONNOUSERSITE=1
PYTHON_EXE=$(which python)

if [[ "$PYTHON_EXE" != *"/envs/$ENV_NAME/"* ]]; then
    echo "❌ ERROR: Activation failed! 'which python' points to: $PYTHON_EXE"
    exit 1
fi
echo "✅ Using Python: $PYTHON_EXE"

# --- Installation ---
echo ">>> Installing Mamba dependencies..."
mamba install -y pandas numpy=1.26.4 ipykernel

echo ">>> Installing PyTorch (GPU) via pip..."
"$PYTHON_EXE" -m pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126

echo ">>> Installing DALI..."
"$PYTHON_EXE" -m pip install nvidia-dali-cuda110==1.50.0

echo ">>> Installing Lightning Pose (Editable Mode)..."
"$PYTHON_EXE" -m pip install -e .

echo "----------------------------------------------------------------"
echo "✅ SUCCESS! Environment '$ENV_NAME' is ready."
echo "To use it, run:"
echo "  module load Mambaforge/23.11.0-fasrc01"
echo "  source activate $ENV_NAME"
echo "  export PYTHONNOUSERSITE=1"
echo "----------------------------------------------------------------"

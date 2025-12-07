#!/bin/bash
# Setup script for Muon optimizer research project

set -e

echo "Muon Optimizer Research Project - Setup Script"
echo "=============================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Detect system type
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    ENV_FILE="environment.yml"
elif command -v nvidia-smi &> /dev/null; then
    echo "Detected system with NVIDIA GPU"
    read -p "Do you want to use CUDA? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ENV_FILE="environment-cuda.yml"
        echo "Using CUDA environment file"
    else
        ENV_FILE="environment.yml"
        echo "Using CPU/MPS environment file"
    fi
else
    echo "Detected system without NVIDIA GPU"
    ENV_FILE="environment.yml"
fi

# Check if environment already exists
if conda env list | grep -q "^muon "; then
    echo "Environment 'muon' already exists."
    read -p "Do you want to remove it and create a new one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n muon
    else
        echo "Keeping existing environment. Activate it with: conda activate muon"
        exit 0
    fi
fi

# Create environment
echo "Creating conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

echo ""
echo "Setup complete!"
echo ""
echo "IMPORTANT: Mac-specific OpenMP fix"
echo "===================================="
echo "To avoid OpenMP crashes on macOS, add this to your ~/.zshrc (or ~/.bashrc):"
echo "  export KMP_DUPLICATE_LIB_OK=TRUE"
echo ""
echo "Or run it in your current shell:"
echo "  export KMP_DUPLICATE_LIB_OK=TRUE"
echo ""

# Check if we're on macOS and add to shell config
if [[ "$OSTYPE" == "darwin"* ]]; then
    SHELL_CONFIG=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    fi
    
    if [ -n "$SHELL_CONFIG" ]; then
        if ! grep -q "KMP_DUPLICATE_LIB_OK" "$SHELL_CONFIG"; then
            read -p "Add KMP_DUPLICATE_LIB_OK=TRUE to $SHELL_CONFIG? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "" >> "$SHELL_CONFIG"
                echo "# Fix for PyTorch OpenMP on macOS" >> "$SHELL_CONFIG"
                echo "export KMP_DUPLICATE_LIB_OK=TRUE" >> "$SHELL_CONFIG"
                echo "Added to $SHELL_CONFIG"
            fi
        else
            echo "KMP_DUPLICATE_LIB_OK already set in $SHELL_CONFIG"
        fi
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     conda activate muon"
echo ""
echo "  2. (macOS only) Set OpenMP fix in current shell:"
echo "     export KMP_DUPLICATE_LIB_OK=TRUE"
echo ""
echo "  3. Verify PyTorch installation:"
echo "     python -c \"import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())\""
echo ""
echo "  4. Run a test training:"
echo "     python -m src.experiments.basic_training --config configs/basic_training_mlp_mnist.yaml"

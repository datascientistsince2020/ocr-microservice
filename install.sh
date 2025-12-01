#!/bin/bash

# OCR Service Installation Script
# Installs all dependencies including system packages

set -e

echo "=========================================="
echo "OCR Service Installation"
echo "=========================================="
echo ""

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"
echo ""

# Install system dependencies
echo "Installing system dependencies..."

if [[ "$OS" == "Linux" ]]; then
    # Linux (Ubuntu/Debian) - Official olmOCR dependencies
    if command -v apt-get &> /dev/null; then
        echo "Using apt-get (installing official olmOCR dependencies)..."
        sudo apt-get update
        sudo apt-get install -y \
            poppler-utils \
            ttf-mscorefonts-installer \
            msttcorefonts \
            fonts-crosextra-caladea \
            fonts-crosextra-carlito \
            gsfonts \
            lcdf-typetools
    elif command -v yum &> /dev/null; then
        echo "Using yum..."
        sudo yum install -y poppler-utils
    else
        echo "⚠️  Could not detect package manager. Please install poppler-utils manually."
    fi

elif [[ "$OS" == "Darwin" ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "Using Homebrew..."
        brew install poppler
    else
        echo "⚠️  Homebrew not found. Please install it from https://brew.sh/"
        echo "Then run: brew install poppler"
        exit 1
    fi

else
    echo "⚠️  Unsupported OS: $OS"
    echo "Please install poppler manually"
fi

echo "✓ System dependencies installed"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."

if [ -d "venv" ]; then
    echo "Virtual environment found, activating..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Detect if CUDA is available for GPU support
echo "Detecting GPU..."
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✓ CUDA detected - Installing GPU version"
    GPU_SUPPORT=true
else
    echo "ℹ️  No CUDA detected - Installing CPU version"
    GPU_SUPPORT=false
fi
echo ""

# Install olmOCR package (includes correct dependencies)
if [ "$GPU_SUPPORT" = true ]; then
    echo "Installing olmOCR with GPU support..."
    pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128

    # Install flash infer for faster GPU inference (recommended)
    echo "Installing flash-infer for faster inference..."
    pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl || \
        echo "⚠️  Flash-infer installation failed (optional, continuing...)"
else
    echo "Installing olmOCR (CPU version)..."
    pip install olmocr
fi

# Install additional requirements from requirements.txt
echo "Installing additional requirements..."
pip install -r requirements.txt

echo "✓ Python dependencies installed"
echo ""

# Check if model should be pre-downloaded
echo "=========================================="
echo "Model Configuration"
echo "=========================================="
echo ""
echo "The service will download the model automatically on first run."
echo "Model: allenai/olmOCR-2-7B-1025-4bit (~3-4GB)"
echo "Location: ~/.cache/huggingface/"
echo ""
read -p "Do you want to pre-download the model now? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pre-downloading model..."
    python3 -c "
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

print('Downloading processor...')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)

print('Downloading model (this may take a few minutes)...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'allenai/olmOCR-2-7B-1025-4bit',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print('✓ Model downloaded successfully!')
"
    echo "✓ Model pre-downloaded"
else
    echo "Skipping pre-download. Model will download on first run."
fi

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To run the service:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run the service: python main.py"
echo ""
echo "Or use Docker:"
echo "  docker-compose up"
echo ""
echo "Configuration: Edit .env file"
echo "Documentation: See README.md"
echo ""

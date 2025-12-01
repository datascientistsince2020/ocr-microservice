# OCR Service - Complete Installation Guide

## Quick Install (Recommended)

```bash
cd ocr-service
./install.sh
```

This automated script will:
✅ Install all system dependencies
✅ Install olmOCR package with GPU support (if CUDA available)
✅ Install flash-infer for faster inference
✅ Optionally pre-download the model
✅ Set up virtual environment

## Manual Installation

### Step 1: System Dependencies

#### Ubuntu/Debian (Linux)
```bash
sudo apt-get update
sudo apt-get install -y \
    poppler-utils \
    ttf-mscorefonts-installer \
    msttcorefonts \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools
```

#### macOS
```bash
brew install poppler
```

### Step 2: Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install olmOCR Package

#### With GPU Support (CUDA)
```bash
pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128

# Recommended: Install flash-infer for faster inference
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
```

#### CPU Only
```bash
pip install olmocr
```

### Step 4: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

print('✓ Imports working')
print(f'CUDA available: {torch.cuda.is_available()}')

processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)
print('✓ Processor loaded')

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'allenai/olmOCR-2-7B-1025-4bit',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
print('✓ Model loaded')
print('✅ All checks passed!')
"
```

## Run the Service

### Development Mode
```bash
source venv/bin/activate
python main.py
```

Service will be available at: `http://localhost:8001`

### Production Mode with Uvicorn
```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
```

## Docker Installation

### Build Image
```bash
docker build -t ocr-service .
```

### Run Container (GPU)
```bash
docker run --gpus all -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e USE_HF=true \
  -e HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit \
  -e DEVICE_TYPE=cuda \
  ocr-service
```

### Run Container (CPU)
```bash
docker run -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e USE_HF=true \
  -e DEVICE_TYPE=cpu \
  ocr-service
```

### Docker Compose
```bash
docker-compose up
```

## Requirements

### Hardware

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 8GB | 16GB | 32GB+ |
| **GPU** | None (CPU mode) | 6GB VRAM | T4/A10G (16GB+) |
| **Disk** | 10GB | 20GB | 50GB+ |

### Software

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.10+ | Required |
| **CUDA** | 12.1+ | For GPU support |
| **PyTorch** | 2.1.1+ | Installed automatically |
| **Transformers** | 4.45.0+ | Installed automatically |

## Model Information

### Default Model
- **Repo**: `allenai/olmOCR-2-7B-1025-4bit`
- **Size**: ~3-4GB (4-bit quantized)
- **Type**: Qwen2.5-VL based
- **Download**: Automatic from HuggingFace

### Model Cache Location
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

### First Run
The model downloads automatically on first run:
- **Time**: 5-10 minutes (depends on internet speed)
- **Space**: ~4GB
- **Subsequent runs**: Instant (uses cache)

## Configuration

Edit `.env` file:

```bash
# Device: auto, mlx, cuda, or cpu
DEVICE_TYPE=auto

# Use HuggingFace auto-download (recommended)
USE_HF=true
HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit

# Or use local model
# USE_HF=false
# MODEL_PATH=/path/to/local/model

# Service port
PORT=8001
```

## Testing

### Health Check
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "model_path": "allenai/olmOCR-2-7B-1025-4bit"
}
```

### Upload & Extract
```bash
# Upload PDF
curl -X POST -F "file=@test.pdf" http://localhost:8001/upload

# Extract page 1 as JSON
curl -X POST "http://localhost:8001/extract/test.pdf" \
  -F "page_number=1" \
  -F "format=json" \
  -F "dpi=50"
```

## Performance Optimization

### GPU Optimization
1. **Flash Infer**: Already installed for faster inference
2. **Batch Size**: Process multiple pages in parallel
3. **DPI**: Lower DPI = faster processing (try 50-100)

### Memory Optimization
1. **4-bit Model**: Already using quantized model
2. **Lower max_tokens**: Reduce from 4096 to 2048
3. **Clear cache**: Service auto-clears after each request

## Troubleshooting

### Issue: "olmocr not found"
```bash
pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128
```

### Issue: "Qwen2_5_VLForConditionalGeneration not found"
```bash
pip install --upgrade transformers>=4.45.0
```

### Issue: "poppler not found"
```bash
# Ubuntu
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

### Issue: CUDA out of memory
**Solutions**:
- Use 4-bit model (already default)
- Lower DPI: `dpi=50`
- Reduce tokens: `max_tokens=2048`
- Use smaller GPU batch size

### Issue: Slow first run
**Expected**: Model downloads on first run (~5-10 mins)
**Solution**: Use `./install.sh` and pre-download the model

### Issue: Flash-infer installation fails
**Solution**: Not critical, service works without it (just slower)
```bash
# Skip flash-infer, continue with installation
pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128
```

## Deployment

### Deploy to Koyeb
```bash
./deploy-koyeb.sh my-ocr-service
```

Environment variables for Koyeb:
```
USE_HF=true
HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit
DEVICE_TYPE=cuda
```

### Deploy to Other Platforms

The Docker image works on:
- AWS ECS/EKS
- Google Cloud Run (with GPU)
- Azure Container Instances
- Railway
- Render
- Any Kubernetes cluster

## Cost Estimates

### Local Development
- **CPU**: Free (slower)
- **GPU**: Requires NVIDIA GPU

### Cloud (Koyeb)
- **GPU (T4)**: ~$0.50-0.70/hour
- **CPU**: ~$0.01-0.02/hour

## Next Steps

1. ✅ Installation complete
2. Test with sample PDF
3. Integrate with your application
4. Deploy to production
5. Monitor performance

## Support

- **Documentation**: [README.md](README.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Deployment**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Fixes**: [FIXES.md](FIXES.md)
- **Official olmOCR**: https://github.com/allenai/olmocr

## Summary

**Quick commands**:
```bash
# Install everything
./install.sh

# Run service
source venv/bin/activate
python main.py

# Test
curl http://localhost:8001/health

# Deploy
./deploy-koyeb.sh my-app
```

Done! 🚀

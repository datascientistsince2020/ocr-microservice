# OCR Service - Issue Fixes

## ✅ Fixed: ImportError for Qwen2_5_VLForConditionalGeneration

### The Issue
```
ImportError: cannot import name 'Qwen2_5_VLForConditionalGeneration' from 'transformers'
```

### Root Cause
The `Qwen2_5_VLForConditionalGeneration` class requires transformers >= 4.45.0, but we had 4.36.0.

### The Fix

Updated [requirements.txt](requirements.txt):
```diff
- transformers==4.36.0
+ transformers>=4.45.0
+ qwen-vl-utils
```

### How to Apply

#### Option 1: Automated Installation (Recommended)
```bash
cd ocr-service
./install.sh
```

This script will:
- Install system dependencies (poppler)
- Create/activate virtual environment
- Install all Python packages
- Optionally pre-download the model

#### Option 2: Manual Installation

1. **Upgrade transformers**:
   ```bash
   pip install --upgrade transformers>=4.45.0
   pip install qwen-vl-utils
   ```

2. **Install system dependencies**:
   ```bash
   # macOS
   brew install poppler

   # Ubuntu/Debian
   sudo apt-get install poppler-utils

   # Or use the install script
   ./install.sh
   ```

3. **Verify installation**:
   ```bash
   python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Success!')"
   ```

## ✅ Fixed: Model Path Error

### The Issue
```
OSError: Incorrect path_or_model_id: '/app/models/olmOCR-2-7B-1025-4bit'
```

### The Fix
Updated default configuration to use HuggingFace automatic download:

**In [.env](.env)**:
```bash
USE_HF=true
HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit
```

**Benefits**:
- No need to manually download models
- Model downloads automatically on first run
- Cached locally for future use
- Works out of the box

## ✅ Fixed: Correct Model Architecture

### Changes Made

1. **Import correct model class**:
   ```python
   from transformers import Qwen2_5_VLForConditionalGeneration
   ```

2. **Use Qwen processor**:
   ```python
   processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
   ```

3. **Load olmOCR weights**:
   ```python
   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
       "allenai/olmOCR-2-7B-1025-4bit",
       torch_dtype=torch.bfloat16,
       trust_remote_code=True
   )
   ```

4. **Updated inference pipeline** to match olmOCR's official usage

## Quick Start (After Fixes)

### 1. Install Dependencies
```bash
cd ocr-service
./install.sh
```

### 2. Run the Service
```bash
# Activate venv (if not already)
source venv/bin/activate

# Run
python main.py
```

**First run**: Model will download (~3-4GB, takes 5-10 mins)
**Subsequent runs**: Instant startup (uses cache)

### 3. Test It
```bash
# Health check
curl http://localhost:8001/health

# Should return:
# {
#   "status": "healthy",
#   "device": "cuda",  # or mlx/cpu
#   "model_loaded": true,
#   "model_path": "allenai/olmOCR-2-7B-1025-4bit"
# }
```

## Docker Setup (After Fixes)

### Build
```bash
docker build -t ocr-service .
```

### Run
```bash
docker run -p 8001:8001 \
  -e USE_HF=true \
  -e HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit \
  -e DEVICE_TYPE=auto \
  ocr-service
```

**Note**: Model will download inside container on first run. Consider pre-downloading and mounting the cache:

```bash
docker run -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e USE_HF=true \
  ocr-service
```

## Requirements Summary

### System Requirements
- Python 3.10+
- poppler-utils (for PDF processing)

### Python Requirements (Key Changes)
- `transformers>=4.45.0` (was 4.36.0)
- `qwen-vl-utils` (new)
- `torch>=2.1.1`
- `accelerate>=0.25.0`

### Hardware Requirements
- **Minimum (CPU)**: 8GB RAM, 10GB disk
- **Recommended (GPU)**: NVIDIA GPU with 6GB+ VRAM
- **Optimal (GPU)**: T4/A10G GPU with 8GB+ VRAM

## Deployment to Koyeb (Updated)

The Dockerfile has been updated with correct dependencies.

### Deploy
```bash
./deploy-koyeb.sh my-ocr-service
```

Or manually set these environment variables in Koyeb:
```
USE_HF=true
HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit
DEVICE_TYPE=cuda
```

**Important**: First deployment will take longer (~10-15 mins) due to model download.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'qwen_vl_utils'"
**Solution**:
```bash
pip install qwen-vl-utils
```

### Issue: "poppler not found"
**Solution**:
```bash
# macOS
brew install poppler

# Ubuntu
sudo apt-get install poppler-utils
```

### Issue: Model download is slow
**Solution**: Be patient on first run. The 4-bit model is ~3-4GB. Subsequent runs use cache.

### Issue: CUDA out of memory
**Solutions**:
- Use 4-bit model (already default)
- Reduce max_tokens in requests
- Lower DPI for smaller images
- Use CPU mode: `DEVICE_TYPE=cpu`

### Issue: Still getting import errors
**Solution**: Ensure you're in the virtual environment and packages are installed:
```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('OK')"
```

## What Changed (Summary)

| Component | Before | After | Reason |
|-----------|--------|-------|--------|
| transformers | 4.36.0 | >=4.45.0 | Qwen2.5-VL support |
| Model loading | AutoModelForVision2Seq | Qwen2_5_VLForConditionalGeneration | Correct class |
| Processor | Generic | Qwen/Qwen2.5-VL-7B-Instruct | Official processor |
| Model source | Local path | HuggingFace auto-download | Ease of use |
| Dependencies | Missing qwen-vl-utils | Added | Required for Qwen models |

## Verification

Run this to verify everything is working:

```bash
cd ocr-service
source venv/bin/activate

python -c "
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

print('✓ Imports working')

processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)
print('✓ Processor loaded')

# This will download the model if not cached
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'allenai/olmOCR-2-7B-1025-4bit',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
print('✓ Model loaded')

print('✅ All checks passed!')
"
```

## Need Help?

- Check [QUICKSTART.md](QUICKSTART.md) for quick setup
- See [README.md](README.md) for full documentation
- Review [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment options

All issues have been resolved! The service is now ready to use. 🚀

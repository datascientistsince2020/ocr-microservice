# OCR Service Quick Start

## Fixed: Model Loading Issue ✅

The service now automatically downloads the 4-bit quantized model from HuggingFace!

**Model**: `allenai/olmOCR-2-7B-1025-4bit`

## Run Locally (No Model Files Needed!)

### Option 1: Run with Python

```bash
cd ocr-service

# Install dependencies
pip install -r requirements.txt

# Run (model downloads automatically on first run)
python main.py
```

The model will be downloaded to your HuggingFace cache (`~/.cache/huggingface/`) on first run.

### Option 2: Run with Docker

```bash
cd ocr-service

# Build
docker build -t ocr-service .

# Run with HuggingFace (recommended)
docker run -p 8001:8001 \
  -e USE_HF=true \
  -e HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit \
  -e DEVICE_TYPE=auto \
  ocr-service
```

### Option 3: Docker Compose

```bash
# Edit docker-compose.yml to use HF model
docker-compose up
```

## Configuration Options

### Use HuggingFace Model (Default - Recommended)

```bash
# .env file
USE_HF=true
HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit
DEVICE_TYPE=auto
```

**Pros:**
- No need to download model manually
- Always up-to-date
- Works out of the box
- Cached locally after first download

### Use Local Model Files

If you have the model downloaded locally:

```bash
# .env file
USE_HF=false
MODEL_PATH=/path/to/your/local/models/olmOCR-2-7B-1025-4bit
DEVICE_TYPE=auto
```

With Docker:
```bash
docker run -p 8001:8001 \
  -v /your/local/models:/app/models \
  -e USE_HF=false \
  -e MODEL_PATH=/app/models/olmOCR-2-7B-1025-4bit \
  ocr-service
```

## Test the Service

Once running, test with:

```bash
# Health check
curl http://localhost:8001/health

# Expected response:
# {
#   "status": "healthy",
#   "device": "cuda",  # or "mlx" or "cpu"
#   "model_loaded": true,
#   "model_path": "allenai/olmOCR-2-7B-1025-4bit"
# }

# Upload a PDF
curl -X POST -F "file=@test.pdf" http://localhost:8001/upload

# Extract text
curl -X POST "http://localhost:8001/extract/test.pdf" \
  -F "page_number=1" \
  -F "format=json" \
  -F "dpi=50"
```

## Deploy to Koyeb

The service is now ready to deploy with automatic model downloading:

```bash
./deploy-koyeb.sh my-ocr-service
```

Or manually in Koyeb dashboard with these environment variables:

```
USE_HF=true
HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-4bit
DEVICE_TYPE=cuda
```

## Hardware Detection

The service automatically detects and uses the best hardware:

- **Apple Silicon (M1/M2/M3)**: Uses MLX backend
- **NVIDIA GPU**: Uses CUDA with fp16
- **CPU**: Falls back to CPU inference

Override with: `DEVICE_TYPE=cuda` or `mlx` or `cpu`

## Model Download Details

### First Run
On first startup, the model (~3-4 GB for 4-bit version) will be downloaded from HuggingFace.

**Download location**:
- Linux/Mac: `~/.cache/huggingface/hub/`
- Windows: `%USERPROFILE%\.cache\huggingface\hub\`

**Time**: 5-10 minutes depending on your internet speed

### Subsequent Runs
The model is loaded from cache instantly (no re-download).

## Troubleshooting

### Issue: Model downloading is slow
**Solution**: The 4-bit model is ~3-4GB. Be patient on first run.

### Issue: Out of memory
**Solutions**:
- Use the 4-bit model (already default)
- Reduce `max_tokens` in requests
- Reduce `dpi` for smaller images
- Use CPU mode if GPU is small

### Issue: Model not found
**Check**:
1. `USE_HF` is set to `true`
2. Internet connection is working
3. HuggingFace is accessible (not blocked)

### Issue: CUDA out of memory on GPU
**Solutions**:
- The 4-bit model uses ~4GB VRAM
- Ensure GPU has at least 6GB VRAM
- Close other GPU applications
- Use smaller images (lower DPI)

## Performance

### 4-bit Quantized Model
- **Speed**: ~90% of full model
- **Quality**: ~95% of full model
- **Memory**: ~25% of full model (3-4GB vs 12-16GB)
- **Recommended**: Yes, great balance

### Inference Speed
- **GPU (T4)**: ~5-10 seconds per page
- **CPU**: ~30-60 seconds per page
- **Apple Silicon (M2)**: ~10-15 seconds per page

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_HF` | `true` | Use HuggingFace model |
| `HF_MODEL_REPO` | `allenai/olmOCR-2-7B-1025-4bit` | Model repo ID |
| `MODEL_PATH` | `models/olmOCR-2-7B-1025-4bit` | Local model path |
| `DEVICE_TYPE` | `auto` | Force device: `auto`, `mlx`, `cuda`, `cpu` |
| `PORT` | `8001` | Service port |
| `HF_TOKEN` | (empty) | HuggingFace token (for private models) |

## Next Steps

1. ✅ Service is running
2. Test with a sample PDF
3. Deploy to Koyeb or your cloud provider
4. Integrate with your main application

See [README.md](README.md) for full API documentation and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment instructions.

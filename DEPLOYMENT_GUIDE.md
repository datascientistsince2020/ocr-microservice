# OCR Service Deployment Guide

## Quick Start

Your OCR service is now ready to deploy! Here's what was created:

### Directory Structure
```
ocr-service/
├── main.py                 # FastAPI application with MLX/CUDA support
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image for CUDA deployment
├── docker-compose.yml     # Local development setup
├── .dockerignore          # Docker build exclusions
├── .env.example           # Environment variables template
├── deploy-koyeb.sh        # Automated Koyeb deployment script
├── README.md              # Full documentation
└── DEPLOYMENT_GUIDE.md    # This file
```

## Features

✅ **Multi-Platform Support**: Automatic detection of MLX (Apple Silicon), CUDA (NVIDIA), or CPU
✅ **Docker Ready**: Pre-configured Dockerfile with GPU support
✅ **Koyeb Compatible**: Ready to deploy to Koyeb with GPU instances
✅ **Production Ready**: Health checks, error handling, and logging
✅ **RESTful API**: Clean API design with FastAPI

## Deployment Options

### Option 1: One-Click Koyeb Deployment (Recommended)

```bash
cd ocr-service
./deploy-koyeb.sh my-ocr-service fra
```

This script will:
1. Create a Koyeb app
2. Deploy with GPU support (NVIDIA T4)
3. Set up health checks
4. Configure environment variables

### Option 2: Manual Koyeb Deployment

1. **Push to GitHub**:
   ```bash
   cd ocr-service
   git init
   git add .
   git commit -m "Initial OCR service"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy via Koyeb Dashboard**:
   - Go to [app.koyeb.com](https://app.koyeb.com)
   - Click "Create Service" → "Deploy from GitHub"
   - Select your repository
   - Configure:
     - Build: Dockerfile
     - Port: 8001
     - Instance: `gpu-nvidia-tesla-t4`
     - Health check: `/health`
     - Environment variables:
       ```
       DEVICE_TYPE=cuda
       MODEL_PATH=/app/models/olmOCR-2-7B-1025-4bit
       ```

### Option 3: Local Docker Development

```bash
cd ocr-service

# With GPU
docker-compose up ocr-service

# CPU only
docker-compose --profile cpu up ocr-service-cpu
```

### Option 4: Local Python Development

```bash
cd ocr-service
pip install -r requirements.txt
export MODEL_PATH=/path/to/your/model
python main.py
```

## Model Setup

### Important: Model Files

Before deployment, you need to handle model files. Choose one option:

#### Option A: Embed in Docker Image (Simple but large)
Add to `Dockerfile` before `CMD`:
```dockerfile
COPY models /app/models
```

#### Option B: Download from HuggingFace on Startup
Add to `Dockerfile`:
```dockerfile
RUN pip install huggingface-hub && \
    python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('YOUR-MODEL-REPO', local_dir='/app/models')"
```

#### Option C: External Storage (Recommended for Production)
Upload model to S3/GCS and download on startup. Add to `main.py`:
```python
@app.on_event("startup")
async def download_model():
    # Download model from S3/GCS
    import boto3
    s3 = boto3.client('s3')
    s3.download_file('bucket', 'model.zip', '/tmp/model.zip')
    # Extract and load
```

## Testing the Deployment

Once deployed, test with:

```bash
# Get your service URL
export OCR_URL="https://your-service.koyeb.app"

# Health check
curl $OCR_URL/health

# Upload PDF
curl -X POST -F "file=@test.pdf" $OCR_URL/upload

# Extract text
curl -X POST "$OCR_URL/extract/test.pdf" \
  -F "page_number=1" \
  -F "format=json" \
  -F "dpi=50"
```

## Integration with Main Backend

Update your main backend (`app/backend/main.py`):

```python
import httpx

# Add environment variable
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://localhost:8001")

# Replace OCR service initialization with API calls
async def extract_with_remote_ocr(filename: str, page: int, format: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Upload file to OCR service
        file_path = UPLOAD_DIR / filename
        with open(file_path, "rb") as f:
            await client.post(
                f"{OCR_SERVICE_URL}/upload",
                files={"file": (filename, f, "application/pdf")}
            )

        # Extract text
        response = await client.post(
            f"{OCR_SERVICE_URL}/extract/{filename}",
            data={
                "page_number": page,
                "format": format,
                "dpi": 50,
                "max_tokens": 4096,
                "temperature": 0.0
            }
        )

        return response.json()["content"]
```

## Cost Estimation (Koyeb)

### GPU Instance (T4)
- **Price**: ~$0.50-0.70/hour
- **Best for**: Production, fast inference
- **Processing**: ~5-10 seconds per page

### CPU Instance (Nano)
- **Price**: ~$0.01-0.02/hour
- **Best for**: Development, low traffic
- **Processing**: ~30-60 seconds per page

**Recommendation**: Start with CPU for testing, upgrade to GPU for production.

## Monitoring

### Health Endpoint
```bash
curl https://your-service.koyeb.app/health
```

Response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "model_path": "/app/models/olmOCR-2-7B-1025-4bit"
}
```

### Koyeb Metrics
- View in dashboard: Requests, latency, errors
- Set up alerts for downtime

### Logging
```bash
koyeb service logs your-app/ocr-api -f
```

## Troubleshooting

### Issue: Model Not Loading
**Solution**:
- Check `MODEL_PATH` environment variable
- Ensure model files are accessible in container
- Check logs: `docker logs ocr-service` or `koyeb service logs`

### Issue: CUDA Out of Memory
**Solutions**:
- Reduce `max_tokens` in requests
- Reduce `dpi` for smaller images
- Upgrade to larger GPU instance
- Use model quantization (4-bit/8-bit)

### Issue: Slow Response Times
**Solutions**:
- Check device type (should be `cuda` not `cpu`)
- Enable caching for repeated requests
- Use batch processing for multiple pages
- Scale horizontally with load balancer

### Issue: Deployment Fails on Koyeb
**Solutions**:
- Check Dockerfile builds locally: `docker build -t ocr-service .`
- Verify all required files are present
- Check Koyeb service logs for errors
- Ensure health check endpoint is responding

## Security Considerations

1. **API Authentication**: Add API key authentication
   ```python
   from fastapi.security import APIKeyHeader

   api_key_header = APIKeyHeader(name="X-API-Key")

   @app.post("/extract/{filename}")
   async def extract_text(api_key: str = Depends(api_key_header)):
       if api_key != os.getenv("API_KEY"):
           raise HTTPException(401)
   ```

2. **Rate Limiting**: Add rate limiting to prevent abuse
   ```python
   from slowapi import Limiter

   limiter = Limiter(key_func=get_remote_address)

   @app.post("/extract/{filename}")
   @limiter.limit("10/minute")
   async def extract_text():
       ...
   ```

3. **File Validation**: Validate uploaded files
   - Check file size limits
   - Verify PDF format
   - Scan for malicious content

## Scaling

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use Redis for shared cache across instances
- Consider queue-based processing for high volume

### Optimization Tips
- Cache extraction results
- Use lower DPI for faster processing
- Implement batch processing endpoint
- Use async workers for concurrent requests

## Next Steps

1. ✅ Deploy OCR service to Koyeb
2. ✅ Test API endpoints
3. ✅ Integrate with main backend
4. ✅ Set up monitoring and alerts
5. ✅ Add authentication if needed
6. ✅ Optimize for production workload

## Support

For issues or questions:
- Check logs: `koyeb service logs app/service`
- Review [README.md](README.md) for detailed API docs
- Open issue in main repository

---

**Ready to deploy?** Run `./deploy-koyeb.sh` to get started! 🚀

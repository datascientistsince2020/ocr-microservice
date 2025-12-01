# OCR Microservice

Standalone OCR service for PDF text extraction using MLX VLM (Apple Silicon) or CUDA (NVIDIA GPUs).

## Features

- Automatic hardware detection (MLX, CUDA, or CPU)
- PDF text extraction with structured JSON or Markdown output
- RESTful API with FastAPI
- Docker support with GPU acceleration
- Designed for deployment on Koyeb, Railway, or any cloud platform

## Hardware Support

The service automatically detects and uses the best available hardware:

- **MLX**: Apple Silicon (M1/M2/M3 Macs)
- **CUDA**: NVIDIA GPUs
- **CPU**: Fallback for any system

You can override detection with the `DEVICE_TYPE` environment variable: `auto`, `mlx`, `cuda`, or `cpu`.

## Local Development

### Prerequisites

- Python 3.10+
- Docker (optional)
- CUDA toolkit (for GPU support)
- Poppler (for PDF processing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
# Set model path
export MODEL_PATH=/path/to/your/model

# Run the service
python main.py
```

The service will be available at `http://localhost:8001`

## Docker Deployment

### Build the Image

```bash
docker build -t ocr-service .
```

### Run with Docker Compose (GPU)

```bash
# GPU version
docker-compose up ocr-service

# CPU version
docker-compose --profile cpu up ocr-service-cpu
```

### Run with Docker (Manual)

```bash
# With GPU
docker run --gpus all -p 8001:8001 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/olmOCR-2-7B-1025-4bit \
  ocr-service

# CPU only
docker run -p 8001:8001 \
  -v $(pwd)/models:/app/models:ro \
  -e DEVICE_TYPE=cpu \
  -e MODEL_PATH=/app/models/olmOCR-2-7B-1025-4bit \
  ocr-service
```

## Deploy to Koyeb

### Option 1: Using Koyeb CLI

1. Install Koyeb CLI:
```bash
curl -fsSL https://cli.koyeb.com/install.sh | bash
```

2. Login:
```bash
koyeb login
```

3. Deploy the service:
```bash
koyeb app create ocr-service

koyeb service create ocr-api \
  --app ocr-service \
  --docker nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  --ports 8001:http \
  --routes /:8001 \
  --env DEVICE_TYPE=cuda \
  --env MODEL_PATH=/app/models/olmOCR-2-7B-1025-4bit \
  --instance-type gpu-nvidia-tesla-t4 \
  --regions fra
```

### Option 2: Using Koyeb Dashboard

1. Go to [Koyeb Dashboard](https://app.koyeb.com/)
2. Click "Create Service"
3. Choose "Docker" as deployment method
4. Configure:
   - **Docker Image**: Build from this repository or use pre-built image
   - **Port**: 8001
   - **Instance Type**: Select GPU instance (e.g., `gpu-nvidia-tesla-t4`)
   - **Environment Variables**:
     - `DEVICE_TYPE=cuda`
     - `MODEL_PATH=/app/models/olmOCR-2-7B-1025-4bit`
   - **Health Check**: `/health`
5. Click "Deploy"

### Option 3: Using GitHub Integration

1. Push this directory to a GitHub repository
2. In Koyeb Dashboard, select "Deploy from GitHub"
3. Choose the repository and branch
4. Configure build settings:
   - **Dockerfile**: `Dockerfile`
   - **Build context**: `/ocr-service`
5. Set environment variables and instance type as above
6. Deploy

### Model Storage for Koyeb

Since Koyeb doesn't have persistent volumes, you have a few options:

#### Option A: Embed Model in Docker Image
```dockerfile
# Add to Dockerfile
COPY models /app/models
```

⚠️ Warning: This will make your Docker image very large (several GB)

#### Option B: Download Model on Startup
```dockerfile
# Add to Dockerfile before CMD
RUN pip install huggingface-hub && \
    python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('your-model-repo', local_dir='/app/models')"
```

#### Option C: Use External Storage (Recommended)
- Upload model to S3, Google Cloud Storage, or similar
- Download on container startup
- Add to startup script in `main.py`

## API Endpoints

### Health Check
```bash
GET /health
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

### Upload PDF
```bash
POST /upload
Content-Type: multipart/form-data

file: <pdf-file>
```

Response:
```json
{
  "filename": "document.pdf",
  "path": "/tmp/ocr-uploads/document.pdf",
  "message": "File uploaded successfully"
}
```

### Extract Text
```bash
POST /extract/{filename}
Content-Type: application/x-www-form-urlencoded

page_number: 1
dpi: 50
format: json
max_tokens: 4096
temperature: 0.0
```

Response:
```json
{
  "filename": "document.pdf",
  "page": 1,
  "format": "json",
  "content": "{ ... extracted content ... }"
}
```

### List Files
```bash
GET /files
```

### Cleanup File
```bash
DELETE /cleanup/{filename}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE_TYPE` | `auto` | Force device: `auto`, `mlx`, `cuda`, or `cpu` |
| `MODEL_PATH` | `models/olmOCR-2-7B-1025-4bit` | Path to OCR model |
| `PORT` | `8001` | Service port |

## Performance Tips

### For CUDA Deployment

1. Use fp16 precision for faster inference:
   - Model automatically loads with `torch.float16` on CUDA
2. Choose appropriate GPU instance:
   - Small models (< 7B): T4 GPU
   - Medium models (7B-13B): A10G GPU
   - Large models (> 13B): A100 GPU

### Memory Management

- The service automatically clears GPU cache after each request
- For high-traffic scenarios, consider batching requests
- Monitor memory usage with `/health` endpoint

## Integration with Main Backend

Update your main backend to call this service:

```python
import httpx

OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://localhost:8001")

async def extract_with_ocr_service(filename: str, page: int, format: str):
    async with httpx.AsyncClient() as client:
        # Upload file
        with open(file_path, "rb") as f:
            upload_response = await client.post(
                f"{OCR_SERVICE_URL}/upload",
                files={"file": f}
            )

        # Extract text
        extract_response = await client.post(
            f"{OCR_SERVICE_URL}/extract/{filename}",
            data={
                "page_number": page,
                "format": format,
                "dpi": 50
            }
        )

        return extract_response.json()
```

## Cost Considerations for Koyeb

- **GPU instances** are more expensive but much faster
- **CPU instances** are cheaper but slower for large documents
- Consider auto-scaling based on traffic
- Use health checks to prevent failed deployments

## Monitoring

Health check endpoint provides:
- Model load status
- Device type (MLX/CUDA/CPU)
- Model path

Add custom monitoring:
```python
# Add to main.py
import prometheus_client
# ... metrics code
```

## Troubleshooting

### Model Not Loading
- Check `MODEL_PATH` is correct
- Ensure model files are accessible
- Check logs: `docker logs ocr-service`

### CUDA Out of Memory
- Reduce `max_tokens` parameter
- Reduce `dpi` for smaller images
- Use smaller model
- Increase GPU instance size

### Slow Performance
- Check device type: Should be `cuda` not `cpu`
- Ensure GPU drivers are installed
- Monitor GPU utilization

## License

This service is part of the Data Readiness by Peche Labs project.

## Support

For issues or questions, please open an issue in the main repository.

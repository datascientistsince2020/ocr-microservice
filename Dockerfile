# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Accept Microsoft fonts EULA non-interactively
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

# Install system dependencies (official olmOCR requirements)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    poppler-utils \
    ttf-mscorefonts-installer \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install olmOCR with GPU support and dependencies
RUN pip3 install --no-cache-dir olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl || true && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create upload directory
RUN mkdir -p /tmp/ocr-uploads

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE_TYPE=auto
ENV USE_HF=true
ENV HF_MODEL_REPO=allenai/olmOCR-2-7B-1025

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8001/health')"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

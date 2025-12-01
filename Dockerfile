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
    curl \
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
# Install PyTorch CUDA 12.1 wheels (match base image)
RUN pip3 install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install olmOCR with GPU support and remaining dependencies (use same CUDA index)
RUN pip3 install --no-cache-dir olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY main.py .

# Create upload directory
RUN mkdir -p /tmp/ocr-uploads

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE_TYPE=auto
ENV USE_HF=true
ENV HF_MODEL_REPO=allenai/olmOCR-2-7B-1025-FP8

# Expose port
EXPOSE 8001

# Health check (very generous timing for model download on first deployment)
# start-period: 15 mins, interval: 15 mins, retries: 3 (total ~45 mins grace period)
HEALTHCHECK --interval=900s --timeout=30s --start-period=900s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

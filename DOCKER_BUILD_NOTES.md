# Docker Build Notes

## Microsoft Fonts EULA

The Dockerfile automatically accepts the Microsoft TrueType Core Fonts EULA during build.

### What This Does

The line:
```dockerfile
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
```

Pre-accepts the EULA for non-interactive installation of:
- Arial
- Courier New
- Times New Roman
- Comic Sans MS
- Georgia
- Impact
- Trebuchet MS
- Verdana
- Webdings

### Why These Fonts?

These fonts are required by olmOCR for proper PDF text rendering and extraction, especially for documents that reference these common fonts.

### License Information

By building this Docker image, you agree to the Microsoft TrueType Core Fonts EULA:
- Reference: http://www.microsoft.com/typography/fontpack/eula.htm
- These fonts are free for distribution but subject to Microsoft's license terms

### Alternative (Without Microsoft Fonts)

If you don't want to accept the Microsoft fonts EULA, you can remove these lines from the Dockerfile:

```dockerfile
# Remove these lines:
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
# And remove this from apt-get install:
ttf-mscorefonts-installer \
```

**Note**: Text extraction quality may be reduced for documents using Microsoft fonts.

## Build Command

```bash
# Standard build
docker build -t ocr-service .

# Build with no cache (clean build)
docker build --no-cache -t ocr-service .

# Build with build args
docker build \
  --build-arg CUDA_VERSION=12.1 \
  -t ocr-service .
```

## Build Time

Expect the first build to take:
- **With cache**: 5-10 minutes
- **Without cache**: 15-20 minutes
- **With model pre-download**: 20-30 minutes

Main time consumers:
1. System package installation: ~2-3 mins
2. PyTorch/transformers installation: ~5-8 mins
3. olmOCR installation: ~3-5 mins
4. Model download (if pre-downloading): ~5-10 mins

## Build Troubleshooting

### Issue: EULA prompt appears

If you see the EULA prompt despite the fix:

```bash
# Ensure debconf-set-selections runs before apt-get
# The Dockerfile should have this order:
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get update && apt-get install -y ttf-mscorefonts-installer
```

### Issue: Font installation fails

```bash
# Check if debconf is available
docker run ubuntu:22.04 which debconf-set-selections

# If missing, add:
RUN apt-get install -y debconf-utils
```

### Issue: Build cache issues

```bash
# Clear Docker cache
docker builder prune -a

# Rebuild from scratch
docker build --no-cache -t ocr-service .
```

## Optimization Tips

### Multi-stage Build (Future)

For smaller images, consider multi-stage builds:

```dockerfile
# Stage 1: Build
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder
# ... install and build ...

# Stage 2: Runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=builder /app /app
```

### Layer Caching

Order matters for build speed:
1. System packages (changes rarely)
2. Python dependencies (changes occasionally)
3. Application code (changes frequently)

Current Dockerfile is already optimized for this.

## Production Considerations

### Image Size

Current image size: ~8-10GB
- CUDA base: ~3GB
- Python packages: ~4-5GB
- System packages: ~1GB
- Model (if included): ~4GB

### Security

```bash
# Scan for vulnerabilities
docker scan ocr-service

# Run as non-root (add to Dockerfile)
RUN useradd -m -u 1000 ocr
USER ocr
```

### Registry

```bash
# Tag for registry
docker tag ocr-service:latest your-registry.com/ocr-service:v1.0

# Push
docker push your-registry.com/ocr-service:v1.0
```

## Build Platforms

### AMD64 (Intel/AMD)
```bash
docker build --platform linux/amd64 -t ocr-service:amd64 .
```

### ARM64 (Graviton, M1/M2)
```bash
docker build --platform linux/arm64 -t ocr-service:arm64 .
```

### Multi-platform
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ocr-service:latest \
  --push \
  .
```

## Summary

✅ Microsoft fonts EULA is pre-accepted
✅ Build is fully automated
✅ No interactive prompts
✅ Optimized layer caching
✅ Production-ready

Just run: `docker build -t ocr-service .`

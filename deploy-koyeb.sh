#!/bin/bash

# Deploy OCR Service to Koyeb
# Usage: ./deploy-koyeb.sh [app-name] [region]

set -e

APP_NAME=${1:-ocr-service}
REGION=${2:-fra}  # fra=Frankfurt, was=Washington, sin=Singapore

echo "=========================================="
echo "Deploying OCR Service to Koyeb"
echo "=========================================="
echo "App Name: $APP_NAME"
echo "Region: $REGION"
echo ""

# Check if koyeb CLI is installed
if ! command -v koyeb &> /dev/null; then
    echo "Error: Koyeb CLI not found. Installing..."
    curl -fsSL https://cli.koyeb.com/install.sh | bash
    echo ""
    echo "Please run 'koyeb login' and then run this script again"
    exit 1
fi

# Check if logged in
if ! koyeb app list &> /dev/null; then
    echo "Error: Not logged in to Koyeb"
    echo "Please run 'koyeb login' first"
    exit 1
fi

echo "Creating Koyeb app: $APP_NAME"
koyeb app create $APP_NAME || echo "App already exists, continuing..."

echo ""
echo "Deploying service with GPU support..."
koyeb service create ocr-api \
  --app $APP_NAME \
  --docker nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  --ports 8001:http \
  --routes /:8001 \
  --env DEVICE_TYPE=cuda \
  --env MODEL_PATH=/app/models/olmOCR-2-7B-1025-4bit \
  --instance-type gpu-nvidia-tesla-t4 \
  --regions $REGION \
  --health-checks http:8001:/health \
  --name ocr-api

echo ""
echo "=========================================="
echo "Deployment initiated!"
echo "=========================================="
echo ""
echo "Check status: koyeb service describe $APP_NAME/ocr-api"
echo "View logs: koyeb service logs $APP_NAME/ocr-api"
echo "Get URL: koyeb service get $APP_NAME/ocr-api --output json | jq -r '.urls[0]'"
echo ""

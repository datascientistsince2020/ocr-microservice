#!/bin/bash

# Setup Git Repository for OCR Service
# This script will create a new git repo and optionally push to GitHub

set -e

echo "=========================================="
echo "Setting up Git Repository for OCR Service"
echo "=========================================="
echo ""

# Check if we're already in a git repo
if [ -d .git ]; then
    echo "⚠️  Git repository already exists in this directory"
    read -p "Do you want to remove it and start fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .git
        echo "✓ Removed existing .git directory"
    else
        echo "Exiting..."
        exit 0
    fi
fi

# Initialize git repo
echo "📦 Initializing Git repository..."
git init
echo "✓ Git repository initialized"
echo ""

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Uploads and cache
uploads/
*.pdf
cache/

# Models (too large for git)
models/
*.bin
*.safetensors
*.gguf

# Docker
docker-compose.override.yml

# Logs
*.log
EOF
    echo "✓ Created .gitignore"
    echo ""
fi

# Add all files
echo "➕ Adding files to git..."
git add .
echo "✓ Files added"
echo ""

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: OCR microservice with MLX/CUDA support

- FastAPI application with automatic hardware detection
- Support for MLX (Apple Silicon) and CUDA (NVIDIA GPUs)
- Docker and docker-compose configuration
- Koyeb deployment scripts
- Complete documentation"
echo "✓ Initial commit created"
echo ""

# Set default branch to main
git branch -M main
echo "✓ Set default branch to 'main'"
echo ""

echo "=========================================="
echo "✅ Local Git repository is ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "Option 1: Create GitHub repo using GitHub CLI (gh)"
echo "  gh repo create ocr-microservice --public --source=. --remote=origin"
echo "  git push -u origin main"
echo ""
echo "Option 2: Create GitHub repo using VS Code"
echo "  1. Open Command Palette (Cmd/Ctrl + Shift + P)"
echo "  2. Type 'Publish to GitHub'"
echo "  3. Follow the prompts"
echo ""
echo "Option 3: Create GitHub repo manually"
echo "  1. Go to https://github.com/new"
echo "  2. Create repository named 'ocr-microservice'"
echo "  3. Run these commands:"
echo "     git remote add origin https://github.com/YOUR_USERNAME/ocr-microservice.git"
echo "     git push -u origin main"
echo ""
echo "Your repository is at: $(pwd)"
echo ""

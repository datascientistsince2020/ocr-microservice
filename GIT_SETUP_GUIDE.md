# Git Repository Setup Guide

This guide shows you how to create a new Git repository for the OCR service and push it to GitHub.

## Quick Start (Automated)

Run the setup script:

```bash
cd ocr-service
./setup-repo.sh
```

This will initialize a git repo, create .gitignore, and make the initial commit.

## Method 1: Using VS Code (Easiest)

### Step 1: Initialize Git Repository

Open the terminal in VS Code (`` Ctrl+` `` or `Cmd+` `):

```bash
cd ocr-service
git init
git add .
git commit -m "Initial commit: OCR microservice"
git branch -M main
```

### Step 2: Publish to GitHub via VS Code

1. **Open Command Palette**: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. **Type**: `Publish to GitHub`
3. **Select**: Choose public or private repository
4. **Name**: `ocr-microservice` (or your preferred name)
5. **Wait**: VS Code will create the repo and push your code

Done! Your code is now on GitHub.

## Method 2: Using GitHub CLI (gh)

### Prerequisites

Install GitHub CLI if you haven't:

```bash
# macOS
brew install gh

# Windows
winget install --id GitHub.cli

# Linux
sudo apt install gh
```

### Setup

```bash
cd ocr-service

# Login to GitHub (one-time)
gh auth login

# Initialize git
git init
git add .
git commit -m "Initial commit: OCR microservice with MLX/CUDA support"
git branch -M main

# Create and push to GitHub
gh repo create ocr-microservice --public --source=. --remote=origin --push

# Or for private repo
gh repo create ocr-microservice --private --source=. --remote=origin --push
```

Your repo will be created at: `https://github.com/YOUR_USERNAME/ocr-microservice`

## Method 3: Manual GitHub Setup

### Step 1: Create Local Repository

```bash
cd ocr-service

# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
venv/
uploads/
models/
*.log
.DS_Store
EOF

# Add and commit
git add .
git commit -m "Initial commit: OCR microservice"
git branch -M main
```

### Step 2: Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `ocr-microservice`
3. Description: "OCR microservice with MLX and CUDA support"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

### Step 3: Push to GitHub

Copy the commands from GitHub (or use these):

```bash
git remote add origin https://github.com/YOUR_USERNAME/ocr-microservice.git
git push -u origin main
```

## Method 4: Using VS Code Source Control Panel

### Step 1: Initialize Repository

1. **Open Source Control**: Click the Source Control icon in VS Code sidebar (or `Cmd+Shift+G`)
2. **Click**: "Initialize Repository"
3. **Stage all changes**: Click the `+` icon next to "Changes"
4. **Commit**: Type a message like "Initial commit" and click the checkmark

### Step 2: Publish to GitHub

1. **Click**: "Publish Branch" button at the bottom
2. **Choose**: Public or Private
3. **Name**: Repository name (e.g., `ocr-microservice`)
4. **Wait**: VS Code uploads your code

Done!

## Verification

After setup, verify your repo:

```bash
# Check remote
git remote -v

# Check status
git status

# View commit history
git log --oneline
```

## Clone to Another Location

To move the code to a completely new directory:

```bash
# After creating the GitHub repo, clone it elsewhere
cd ~/Projects  # or wherever you want
git clone https://github.com/YOUR_USERNAME/ocr-microservice.git
cd ocr-microservice
```

## Important: .gitignore

Make sure your `.gitignore` includes:

```gitignore
# Don't commit these
__pycache__/
*.pyc
venv/
.env
uploads/
cache/

# Models are too large for git
models/
*.bin
*.safetensors
*.gguf

# Sensitive
.env
.env.local
```

## Model Files

⚠️ **Important**: Model files are typically too large for Git (>100MB).

Options:
1. **Git LFS** (Git Large File Storage)
2. **External storage** (S3, Hugging Face)
3. **Download on deployment** (recommended)

### Using Git LFS for Models (Optional)

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or download from https://git-lfs.github.com/

# Initialize
git lfs install

# Track model files
git lfs track "models/**/*.bin"
git lfs track "models/**/*.safetensors"

# Add .gitattributes
git add .gitattributes

# Commit
git commit -m "Add Git LFS tracking for model files"
git push
```

## Working with Multiple Repositories

If you want to keep this separate from your main project:

```bash
# In your main project
cd ~/Experiments-/

# Create a submodule (links to OCR repo)
git submodule add https://github.com/YOUR_USERNAME/ocr-microservice.git ocr-service

# Or just reference it in documentation
# Keep them as separate repos
```

## Next Steps After Setup

1. **Update README.md**: Add your GitHub repo URL
2. **Add CI/CD**: GitHub Actions for testing
3. **Deploy**: Use deploy-koyeb.sh or connect to Koyeb directly
4. **Documentation**: Update with your deployment URL

## Troubleshooting

### Issue: "Repository already exists"
```bash
# Remove existing git
rm -rf .git
# Start over with git init
```

### Issue: "Large files detected"
```bash
# Remove large files from history
git rm --cached models/*.bin
# Or use Git LFS (see above)
```

### Issue: "Authentication failed"
```bash
# For HTTPS, use personal access token
# Go to: GitHub → Settings → Developer settings → Personal access tokens

# Or use SSH
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add to GitHub: Settings → SSH and GPG keys
```

## Summary

**Recommended for VS Code users**: Method 1 (VS Code Publish to GitHub)
- Easiest and fastest
- Built into VS Code
- Handles authentication automatically

**Recommended for CLI users**: Method 2 (GitHub CLI)
- Quick and powerful
- Great for automation
- One command to create and push

Choose the method that fits your workflow best!

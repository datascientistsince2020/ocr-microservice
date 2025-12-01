from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import gc
from typing import Optional
import platform
import sys
import time

# Detect hardware and import appropriate libraries
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "auto")  # auto, mlx, cuda

def detect_device():
    """Auto-detect available hardware"""
    if DEVICE_TYPE != "auto":
        return DEVICE_TYPE

    # Check if MLX is available (Apple Silicon)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core as mx
            return "mlx"
        except ImportError:
            pass

    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Fallback to CPU with transformers
    return "cpu"

DEVICE = detect_device()
print(f"[STARTUP] Detected device: {DEVICE}")

# Import appropriate libraries based on device
if DEVICE == "mlx":
    import mlx_vlm
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    print("[STARTUP] Using MLX backend for Apple Silicon")
elif DEVICE == "cuda":
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print("[STARTUP] Using CUDA backend for NVIDIA GPU")
else:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print("[STARTUP] Using CPU backend with transformers")

app = FastAPI(
    title="OCR Microservice",
    description="Standalone OCR service for PDF text extraction (MLX/CUDA/CPU)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("/tmp/ocr-uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Model configuration - supports local path or HuggingFace repo
MODEL_PATH = os.getenv("MODEL_PATH", "models/olmOCR-2-7B-1025-4bit")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "allenai/olmOCR-2-7B-1025-4bit")  # 4-bit quantized model
USE_HF = os.getenv("USE_HF", "true").lower() == "true"  # Default to HuggingFace

# Global OCR service instance
ocr_model = None
ocr_processor = None
ocr_config = None


class ExtractRequest(BaseModel):
    page_number: int
    dpi: int = 50
    format: str = "json"
    max_tokens: int = 512
    temperature: float = 0.0


@app.on_event("startup")
async def startup_event():
    """Load OCR model on startup"""
    global ocr_model, ocr_processor, ocr_config

    # Determine model source
    if USE_HF:
        model_source = HF_MODEL_REPO
        print(f"[STARTUP] Using HuggingFace model: {model_source}")
        print(f"[INFO] Model will be downloaded and cached automatically")
    else:
        model_source = MODEL_PATH
        print(f"[STARTUP] Using local model: {model_source}")

        # Check if local model exists
        if not Path(model_source).exists():
            print(f"[ERROR] Local model not found at: {model_source}")
            print(f"[INFO] Please either:")
            print(f"  1. Mount model volume: -v /path/to/models:/app/models")
            print(f"  2. Use HuggingFace: -e USE_HF=true -e HF_MODEL_REPO=repo-name")
            print(f"  3. Set correct MODEL_PATH: -e MODEL_PATH=/correct/path")
            raise FileNotFoundError(f"Model not found at {model_source}")

    try:
        if DEVICE == "mlx":
            # Load with MLX
            ocr_model, ocr_processor = mlx_vlm.load(model_source)
            ocr_config = load_config(model_source)
            print("[STARTUP] OCR model loaded successfully with MLX")

        elif DEVICE == "cuda":
            # Load with transformers on CUDA using Qwen2.5-VL
            print(f"[STARTUP] Loading processor from Qwen/Qwen2.5-VL-7B-Instruct...")
            ocr_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True
            )

            print(f"[STARTUP] Loading model from {model_source}...")
            ocr_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            print("[STARTUP] OCR model loaded successfully with CUDA")

        else:
            # Load with transformers on CPU using Qwen2.5-VL
            print(f"[STARTUP] Loading processor from Qwen/Qwen2.5-VL-7B-Instruct...")
            ocr_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True
            )

            print(f"[STARTUP] Loading model from {model_source}...")
            ocr_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                trust_remote_code=True
            ).eval()
            print("[STARTUP] OCR model loaded successfully with CPU")

    except Exception as e:
        print(f"[ERROR] Failed to load OCR model: {str(e)}")
        print(f"[ERROR] Model source: {model_source}")
        print(f"[ERROR] Device: {DEVICE}")
        raise


@app.get("/")
async def root():
    return {
        "service": "OCR Microservice",
        "status": "running",
        "model_loaded": ocr_model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Koyeb"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": ocr_model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing"""
    print(f"[UPLOAD] Received file: {file.filename}")

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[UPLOAD] File saved: {file_path}")

    return {
        "filename": file.filename,
        "path": str(file_path),
        "message": "File uploaded successfully"
    }


def extract_pdf_page(pdf_path: str, page_number: int, dpi: int = 72, max_size: int = 896):
    """Extract PDF page with aggressive size reduction"""
    pages = convert_from_path(pdf_path, dpi=dpi)
    if page_number < 1 or page_number > len(pages):
        raise ValueError(f"PDF has {len(pages)} pages. You asked for: {page_number}")

    image = pages[page_number - 1]

    # Resize to safe size
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)

    print(f"[OCR] Image size: {image.size} (DPI: {dpi})")
    return image


def ocr_pdf_page(pdf_path: str, page_number: int, prompt: str,
                 max_tokens: int = 4096, temperature: float = 0.0, dpi: int = 50):
    """Perform OCR on PDF page"""
    global ocr_model, ocr_processor, ocr_config

    if not ocr_model or not ocr_processor:
        raise HTTPException(status_code=500, detail="OCR model not loaded")

    # Enforce token limits (keep small on CPU to avoid long Koyeb timeouts)
    token_cap = 512 if DEVICE != "cuda" else 2048
    max_tokens = max(64, min(max_tokens, token_cap))
    print(f"[OCR] Starting generation | device={DEVICE} | max_tokens={max_tokens} | dpi={dpi}")
    start_time = time.time()

    # Extract image
    image = extract_pdf_page(pdf_path, page_number, dpi=dpi, max_size=896)

    try:
        if DEVICE == "mlx":
            # MLX backend
            formatted_prompt = apply_chat_template(
                ocr_processor,
                ocr_config,
                prompt,
                num_images=1
            )

            output = generate(
                ocr_model,
                ocr_processor,
                formatted_prompt,
                [image],
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False
            )

            # Extract text
            if hasattr(output, 'text'):
                output_text = output.text
            else:
                output_text = str(output)
            print(f"[OCR] Output (mlx) preview: {str(output_text)[:400]}")

        else:
            # CUDA/CPU backend with Qwen2.5-VL
            # Format messages for chat template (following olmOCR example)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"}
                    ]
                }
            ]

            # Apply chat template
            text = ocr_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = ocr_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            # Move to device
            device = torch.device("cuda" if DEVICE == "cuda" else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                output = ocr_model.generate(
                    **inputs,
                    temperature=max(temperature, 0.1),  # Min temp 0.1
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0
                )

            # Decode output (skip prompt tokens)
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            output_text = ocr_processor.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )[0]
            duration = time.time() - start_time
            print(f"[OCR] Generation complete in {duration:.2f}s | output_tokens={new_tokens.shape[1]}")
            print(f"[OCR] Output (qwen) preview: {output_text[:400]}")

    finally:
        # Cleanup
        del image
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if 'start_time' in locals():
            total = time.time() - start_time
            print(f"[OCR] Cleanup complete | total_time={total:.2f}s")

    return output_text


@app.post("/extract/{filename}")
async def extract_text(
    filename: str,
    page_number: int = Form(...),
    dpi: int = Form(50),
    format: str = Form("json"),
    max_tokens: int = Form(512),
    temperature: float = Form(0.0)
):
    """Extract text from a specific PDF page"""
    print(f"[EXTRACT] Request - File: {filename}, Page: {page_number}, Format: {format}")

    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        if format == "json":
            prompt = """Extract ALL text from this document page with ABSOLUTE COMPLETENESS and STRICT HIERARCHY TRACKING.

CRITICAL EXTRACTION RULES - DO NOT SKIP ANYTHING:
1. Extract EVERY SINGLE word, number, symbol, and character EXACTLY as shown - ZERO TOLERANCE for missing content
2. Handle ALL content types regardless of format:
   - Regular text (any size, font, color, orientation)
   - Rotated, angled, or vertical text
   - Text in shapes (circles, boxes, arrows, callouts)
   - Watermarks and background text
   - Headers and footers (even if small or faded)
   - Annotations, comments, and margin notes
   - Mathematical symbols and equations
   - Special characters and Unicode symbols
3. TABLES ARE CRITICAL: Extract ALL table data with complete structure:
   - Capture EVERY cell, row, and column
   - Handle merged cells and complex layouts
   - Preserve headers and data relationships
   - Use "table" type with "rows" array containing all cells
   - Include column headers separately
   - Extract tables even without borders (aligned columns)
4. CHARTS AND VISUAL ELEMENTS: Describe and extract data:
   - Chart type (bar, line, pie, scatter, etc.)
   - All visible labels, legends, and values
   - Axis labels and tick marks
   - Data points and their values
   - Any annotations or callouts on charts
   - Use "chart" type with "chart_type" and "data" fields
5. Track content flow with explicit tags:
   - "section": Main sections/divisions
   - "heading": Headers and titles (include level: 1, 2, 3)
   - "subheading": Subsections
   - "paragraph": Body text blocks
   - "list_item": Bullet or numbered list items
   - "table": Tabular data (MUST include ALL rows and columns)
   - "chart": Charts, graphs, diagrams
   - "callout": Text in shapes, boxes, callouts
   - "annotation": Margin notes, comments
   - "footer": Footer text
   - "header": Header text
   - "watermark": Watermark text
   - "metadata": Dates, page numbers, document info
   - "caption": Figure/table captions
   - "equation": Mathematical equations
6. Include "order" field (1, 2, 3...) showing EXACT reading sequence from top to bottom, left to right
7. Include "level" field for nested hierarchy (0=top level, 1=nested, 2=deeply nested, etc.)
8. Return ONLY valid JSON with complete structure showing ALL content
9. NO markdown code blocks, NO extra commentary, NO truncation

Example structure:
{
  "page": 1,
  "content": [
    {"order": 1, "type": "heading", "level": 1, "text": "Document Title"},
    {"order": 2, "type": "paragraph", "level": 0, "text": "Full paragraph text..."},
    {"order": 3, "type": "table", "level": 0, "columns": 3,
     "headers": ["Column1", "Column2", "Column3"],
     "rows": [
       ["Data1", "Data2", "Data3"],
       ["Data4", "Data5", "Data6"]
     ]}
  ]
}"""

        elif format == "markdown":
            prompt = """Extract ALL text from this document page with ABSOLUTE COMPLETENESS and STRICT HIERARCHY TRACKING in Markdown format.

CRITICAL EXTRACTION RULES:
1. Extract EVERY word, number, symbol exactly as shown
2. Use proper markdown syntax:
   - # Header 1, ## Header 2, ### Header 3 for headings
   - **bold** for emphasis
   - * for list items
   - Proper markdown tables with | separators
3. Add inline tags for special content:
   - [TABLE: description] before tables
   - [CHART: description] for charts
   - [CALLOUT: text] for text in shapes
   - [ANNOTATION: text] for margin notes
   - [HEADER] and [FOOTER] for headers/footers
4. Extract ALL tables with complete structure
5. Maintain exact reading order
6. NO code blocks around output, NO extra commentary

Example:
# [SECTION: Document Title]

## Main Section

Full paragraph text here...

[TABLE: Data Summary]
| Header1 | Header2 | Header3 |
|---------|---------|---------|
| Data1   | Data2   | Data3   |"""

        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use: json or markdown")

        result = ocr_pdf_page(
            str(file_path),
            page_number,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            dpi=dpi
        )

        print(f"[EXTRACT] Successfully extracted page {page_number}")

        return {
            "filename": filename,
            "page": page_number,
            "format": format,
            "content": result
        }

    except Exception as e:
        print(f"[ERROR] Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup/{filename}")
async def cleanup_file(filename: str):
    """Delete uploaded file"""
    file_path = UPLOAD_DIR / filename

    if file_path.exists():
        os.remove(file_path)
        return {"message": "File deleted successfully"}

    raise HTTPException(status_code=404, detail="File not found")


@app.get("/files")
async def list_files():
    """List all uploaded files"""
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"files": files, "count": len(files)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

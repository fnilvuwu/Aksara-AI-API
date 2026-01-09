import os
import uvicorn
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from processor.ProcessorAksaraLontara import ProcessorAksaraLontara

# ===========================================
# Ensure directories exist
# ===========================================
IMAGE_DIR = "dir_images"
PDF_DIR = "dir_pdf"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# ===========================================
# Initialize processor
# ===========================================
processor = ProcessorAksaraLontara(
    ocr_model_path=r"dir_ocr_models\PP-OCRv5_server_rec_infer\buginese_ocr_model.onnx",
    ocr_dict_path=r"dir_ocr_models\PP-OCRv5_server_rec_infer\lontara_chr.txt"
)
# ===========================================
# Main API
# ===========================================
app = FastAPI(
    title="Aksara-AI-API",
    description="Unified API for Aksara ID AI Tools — supporting multi-aksara OCR, translation, and recognition.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================
# Request Models
# ===========================================
class TextRequest(BaseModel):
    text: str

# ===========================================
# Router for Aksara Lontara
# ===========================================
router_lontara = APIRouter(
    prefix="/lontara",
    tags=["Aksara Lontara"]
)

# -------------------------------------------------------------
# 1. Translate Text (Lontara)
# -------------------------------------------------------------
@router_lontara.post("/translate/text")
async def translate_text(payload: TextRequest):
    try:
        result = processor.generate_translation_from_text(payload.text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# 2. Translate Image (Lontara)
# -------------------------------------------------------------
@router_lontara.post("/translate/image")
async def translate_image(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(IMAGE_DIR, f"{timestamp}_{file.filename}")

        # Save file
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        result = processor.generate_translation_from_image(save_path)

        return {"success": True, "file_saved": save_path, "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# 3. Translate PDF (Lontara)
# -------------------------------------------------------------
@router_lontara.post("/translate/pdf")
async def translate_pdf(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(PDF_DIR, f"{timestamp}_{file.filename}")

        # Save file
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        result = processor.generate_translation_from_pdf(save_path)

        return {"success": True, "file_saved": save_path, "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Register Routers
# ===========================================
app.include_router(router_lontara)

# ===========================================
# Root endpoint
# ===========================================
@app.get("/")
def home():
    return {
        "message": "Aksara-AI-API is running.",
        "supported_aksara": [
            "Lontara (active)",
            "Other aksara coming soon: Mandailing, Toba, Lampung, Rencong, Karo..."
        ]
    }


# ===========================================
# Run App
# ===========================================
if __name__ == "__main__":
    uvicorn.run("app:app", port=8080, reload=True)
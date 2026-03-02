"""
routes/vision.py
----------------
Computer Vision endpoints.
Accepts uploaded medical images (JPG / PNG) and returns:
  - Detected abnormality label (e.g., Pneumonia, Normal)
  - Confidence score

Delegates processing to services/vision_service.py.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.vision_service import analyze_image
from app.utils.file_utils import validate_image_file

router = APIRouter()


@router.post("/analyze-image", summary="Analyze a medical image using Computer Vision")
async def analyze_medical_image(file: UploadFile = File(...)):
    """
    Upload a medical image (JPG or PNG).
    Returns predicted condition and confidence score.
    """
    validate_image_file(file)

    contents = await file.read()
    result = analyze_image(contents, file.filename)
    return result


@router.get("/supported-formats", summary="List supported image formats")
async def supported_image_formats():
    """Returns image formats supported by the CV module."""
    return {"formats": ["jpg", "jpeg", "png"]}

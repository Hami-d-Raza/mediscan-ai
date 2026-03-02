"""
routes/mri.py
-------------
Brain MRI analysis endpoint for MediScan AI.

Route:
    POST /analyze-brain-mri
"""

import os
import uuid
import shutil
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import settings
from app.services.mri_model import predict_brain_mri

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png"}


def _save_temp_file(file: UploadFile) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "bin"
    dest_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return dest_path


def _delete_temp_file(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Could not delete temp file '%s': %s", path, exc)


@router.post(
    "/analyze-brain-mri",
    summary="Analyze a Brain MRI image",
    status_code=status.HTTP_200_OK,
)
async def analyze_brain_mri(file: UploadFile = File(...)):
    """
    Upload a Brain MRI image (JPG or PNG) for tumour classification.
    Uses the dedicated MobileNetV2 model trained on the Brain Tumor MRI Dataset.

    Returns:
        prediction, cancer_type, confidence, class_index, simulated, description
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Only JPG and PNG accepted.",
        )

    try:
        saved_path = _save_temp_file(file)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not save uploaded file.") from exc

    try:
        result = predict_brain_mri(saved_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Uploaded file missing during analysis.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Image could not be processed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error during Brain MRI inference: %s", exc)
        raise HTTPException(status_code=500, detail="Unexpected error during analysis.") from exc
    finally:
        _delete_temp_file(saved_path)

    return {
        "filename":    file.filename,
        "prediction":  result["prediction"],
        "cancer_type": result["cancer_type"],
        "confidence":  result["confidence"],
        "class_index": result["class_index"],
        "simulated":   result["simulated"],
        "description": result["description"],
    }

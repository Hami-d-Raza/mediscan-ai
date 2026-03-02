"""
routes/image.py
---------------
Image analysis endpoint for MediScan AI.
Accepts a medical image (JPG/PNG), validates the file type,
saves it temporarily to disk, runs inference via image_model.py,
then deletes the temp file before returning the result.

Route:
    POST /analyze-image
"""

import os
import uuid
import shutil
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import settings
from app.services.image_model import predict_image

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed MIME types
ALLOWED_TYPES = {"image/jpeg", "image/png"}


def _save_temp_file(file: UploadFile) -> str:
    """
    Persist the uploaded file to the configured upload directory.
    Returns the path where the file was saved.
    """
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "bin"
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    dest_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return dest_path


def _delete_temp_file(path: str) -> None:
    """Silently remove a temporary file; logs a warning if deletion fails."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.debug("Deleted temp file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete temp file '%s': %s", path, exc)


@router.post(
    "/analyze-image",
    summary="Analyze a medical image",
    status_code=status.HTTP_200_OK,
)
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload a medical image (JPG or PNG) for diagnostic analysis.

    **Pipeline:**
    1. Validate MIME type → `415` if unsupported.
    2. Save file temporarily to `uploads/`.
    3. Run ResNet18 inference via `predict_image()`.
    4. Delete the temp file (success **or** failure).
    5. Return structured result.

    **Returns:**
    ```json
    {
        "filename":    "chest_xray.jpg",
        "prediction":  "Pneumonia",
        "confidence":  0.9103,
        "class_index": 1,
        "simulated":   false
    }
    ```
    """
    # -----------------------------------------------------------------------
    # 1. Validate file type
    # -----------------------------------------------------------------------
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Only JPG and PNG images are accepted."
            ),
        )

    # -----------------------------------------------------------------------
    # 2. Save file temporarily
    # -----------------------------------------------------------------------
    try:
        saved_path = _save_temp_file(file)
    except OSError as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the uploaded file. Please try again.",
        ) from exc

    # -----------------------------------------------------------------------
    # 3. Run inference — always clean up the temp file afterwards
    # -----------------------------------------------------------------------
    try:
        result = predict_image(saved_path)

    except FileNotFoundError as exc:
        logger.error("Temp file missing before inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uploaded file was not found when running analysis.",
        ) from exc

    except ValueError as exc:
        logger.warning("Invalid image content for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image could not be processed: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during inference for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during image analysis.",
        ) from exc

    finally:
        # 4. Delete temp file — runs whether inference succeeded or raised
        _delete_temp_file(saved_path)

    # -----------------------------------------------------------------------
    # 5. Return real model output
    # -----------------------------------------------------------------------
    return {
        "filename":    file.filename,
        "prediction":  result["prediction"],
        "cancer_type": result["cancer_type"],
        "confidence":  result["confidence"],
        "class_index": result["class_index"],
        "simulated":   result["simulated"],
    }

"""
services/vision_service.py
--------------------------
Legacy Computer Vision stub — NOT used by the main analysis pipeline.

The active multi-cancer and Brain MRI pipelines live in:
  services/image_model.py   → POST /analyze-image
  services/mri_model.py     → POST /analyze-brain-mri

This file is kept for the /vision/analyze-image route (prefixed under /vision)
but is not called by the frontend.
"""

import io
from typing import Any


def _preprocess_image(content: bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(content)).convert("RGB")
    return img


def analyze_image(content: bytes, filename: str) -> dict[str, Any]:
    """
    Stub endpoint — returns a placeholder response.
    Use POST /analyze-image (image_model.py) for real inference.
    """
    return {
        "filename": filename,
        "prediction": "N/A",
        "confidence": 0.0,
        "note": "This endpoint is a placeholder. Use POST /analyze-image for real cancer classification.",
    }

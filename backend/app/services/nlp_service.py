"""
services/nlp_service.py
-----------------------
Orchestration layer for NLP analysis in MediScan AI.

Responsibilities:
  - Accept raw file bytes + filename from the route layer.
  - Extract plain text (PDF or TXT) via file_handler.
  - Delegate entity extraction to nlp_model.analyze_report().
  - Return a unified response dict.
"""

import io
import logging
from typing import Any

from app.utils.file_handler import extract_text
from app.services.nlp_model import analyze_report as _nlp_analyze

logger = logging.getLogger(__name__)


def analyze_report(content: bytes, filename: str) -> dict[str, Any]:
    """
    Main entry point called by the route layer.

    Args:
        content:  Raw bytes of the uploaded file.
        filename: Original filename (used to determine file type).

    Returns:
        {
            "filename":    str,
            "char_count":  int,
            "entities": {
                "diseases":    [...],
                "symptoms":    [...],
                "medications": [...],
                "other":       [...],
            },
            "summary":    str,
            "simulated":  bool,
        }
    """
    # --- Determine MIME type from extension for file_handler ---------------
    ext_to_mime = {".pdf": "application/pdf", ".txt": "text/plain"}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = ext_to_mime.get(ext, "text/plain")

    # --- Extract raw text --------------------------------------------------
    try:
        raw_text = extract_text(content, mime, filename)
    except Exception as exc:
        logger.warning("Text extraction error for '%s': %s", filename, exc)
        raw_text = content.decode("utf-8", errors="replace")

    # --- NLP entity extraction ---------------------------------------------
    result = _nlp_analyze(raw_text)

    return {
        "filename":   filename,
        "char_count": result["char_count"],
        "entities": {
            "diseases":    result["diseases"],
            "symptoms":    result["symptoms"],
            "medications": result["medications"],
            "other":       result.get("other", []),
        },
        "suggested_medications": result.get("suggested_medications", []),
        "summary":   result["summary"],
        "simulated": result["simulated"],
    }

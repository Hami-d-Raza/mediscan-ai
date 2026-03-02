"""
routes/report.py
----------------
Medical report analysis endpoint for MediScan AI.

Accepts plain text or PDF files, extracts the raw text, and returns
structured medical entities extracted by the NLP layer.

Route:
    POST /analyze-report
"""

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.utils.file_handler import SUPPORTED_REPORT_TYPES
from app.services.nlp_service import analyze_report

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analyze-report",
    summary="Analyze a medical report",
    status_code=status.HTTP_200_OK,
)
async def analyze_medical_report(file: UploadFile = File(...)):
    """
    Upload a plain-text or PDF medical report for NLP analysis.

    **Pipeline:**
    1. Validate MIME type → `415` if unsupported.
    2. Read file bytes.
    3. Run NLP entity extraction via `nlp_service.analyze_report()`.
    4. Return structured medical entities.

    **Returns:**
    ```json
    {
        "filename":   "report.pdf",
        "char_count": 1024,
        "entities": {
            "diseases":    ["Diabetes"],
            "symptoms":    ["Fever", "Cough"],
            "medications": ["Paracetamol"],
            "other":       []
        },
        "summary":   "...",
        "simulated": false
    }
    ```
    """
    # -----------------------------------------------------------------------
    # 1. Validate MIME type
    # -----------------------------------------------------------------------
    if file.content_type not in SUPPORTED_REPORT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {sorted(SUPPORTED_REPORT_TYPES)}"
            ),
        )

    # -----------------------------------------------------------------------
    # 2. Read file bytes
    # -----------------------------------------------------------------------
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the uploaded file. Please try again.",
        ) from exc

    # -----------------------------------------------------------------------
    # 3. NLP analysis
    # -----------------------------------------------------------------------
    try:
        result = analyze_report(contents, file.filename)
    except ValueError as exc:
        logger.warning("NLP validation error for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("NLP runtime error for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLP service is currently unavailable. Please try again later.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected NLP error for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during report analysis.",
        ) from exc

    # -----------------------------------------------------------------------
    # 4. Return structured response
    # -----------------------------------------------------------------------
    return result


@router.get("/analyze-report/formats", summary="List supported report formats")
async def supported_formats():
    """Returns accepted file formats for report upload."""
    return {"supported_types": sorted(SUPPORTED_REPORT_TYPES)}

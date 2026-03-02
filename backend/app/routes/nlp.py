"""
routes/nlp.py
-------------
NLP analysis endpoints.
Accepts uploaded medical reports (PDF / plain text) and returns:
  - Extracted medical entities (diseases, symptoms, medications)
  - Structured report summary

Delegates processing to services/nlp_service.py.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.nlp_service import analyze_report
from app.utils.file_utils import validate_report_file

router = APIRouter()


@router.post("/analyze-report", summary="Analyze a medical report using NLP")
async def analyze_medical_report(file: UploadFile = File(...)):
    """
    Upload a medical report (PDF or .txt).
    Returns extracted entities and a structured summary.
    """
    validate_report_file(file)

    contents = await file.read()
    result = analyze_report(contents, file.filename)
    return result


@router.get("/supported-formats", summary="List supported report formats")
async def supported_formats():
    """Returns file formats supported by the NLP module."""
    return {"formats": ["pdf", "txt"]}

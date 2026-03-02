"""
models/analysis.py
------------------
Pydantic response schemas for NLP and Computer Vision analysis results.
Used by FastAPI to validate and document API responses automatically.
"""

from pydantic import BaseModel
from typing import Optional


class NLPEntities(BaseModel):
    diseases:    list[str]
    symptoms:    list[str]
    medications: list[str]
    findings:    list[str]


class NLPAnalysisResponse(BaseModel):
    """Response schema returned by POST /nlp/analyze-report."""
    filename:   str
    char_count: int
    entities:   NLPEntities
    summary:    str


class CVAnalysisResponse(BaseModel):
    """Response schema returned by POST /vision/analyze-image."""
    filename:   str
    prediction: str
    confidence: float
    note:       Optional[str] = None

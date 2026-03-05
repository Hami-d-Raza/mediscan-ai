"""
config.py
---------
Central configuration for MediScan AI.
Reads environment variables (with sensible defaults) using Pydantic Settings.
Import `settings` anywhere in the app to access configuration values.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

# Absolute path to the backend/ directory, works regardless of cwd
_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------
    APP_NAME: str = "MediScan AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # -----------------------------------------------------------------------
    # Server
    # -----------------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        # Production — Azure backend self-reference
        "https://mediscan-api-dqcxaaaxc6aqccgd.southeastasia-01.azurewebsites.net",
        # Production — Vercel + custom domain
        "https://mediscanai.me",
        "https://www.mediscanai.me",
        "https://mediscan-ai.vercel.app",
    ]

    # -----------------------------------------------------------------------
    # Security
    # -----------------------------------------------------------------------
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # -----------------------------------------------------------------------
    # File Upload
    # -----------------------------------------------------------------------
    UPLOAD_DIR: str = str(_BACKEND_DIR / "uploads")
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png"]
    ALLOWED_REPORT_TYPES: List[str] = ["application/pdf", "text/plain"]

    # -----------------------------------------------------------------------
    # ML Models
    # -----------------------------------------------------------------------
    NLP_MODEL_NAME: str = "d4data/biomedical-ner-all"
    CV_MODEL_PATH: str   = str(_BACKEND_DIR / "models" / "multi_cancer_classifier.pt")
    CV_LABELS_PATH: str  = str(_BACKEND_DIR / "models" / "class_labels.json")
    MRI_MODEL_PATH: str  = str(_BACKEND_DIR / "models" / "mri_classifier.pt")
    # Binary classifier: medical (MRI/histo) vs non-medical images.
    # Train offline and place the weights here.  Without this file the
    # system uses the HuggingFace API fallback (Tier-2).
    MEDICAL_CLASSIFIER_PATH: str = str(_BACKEND_DIR / "models" / "medical_classifier.pt")

    # -----------------------------------------------------------------------
    # Medical Image Validation
    # Set True locally to enable Tier-1/Tier-2 validation.
    # Leave False (default) on production — validation is skipped entirely.
    # -----------------------------------------------------------------------
    ENABLE_MEDICAL_VALIDATION: bool = False

    # -----------------------------------------------------------------------
    # HuggingFace Inference API — Tier-2 medical/non-medical fallback
    # Get a free token at https://huggingface.co/settings/tokens
    # Leave blank to skip the API check (neutral score 0.5 used instead).
    # -----------------------------------------------------------------------
    HF_API_TOKEN: str = ""
    HF_MEDICAL_MODEL: str = "openai/clip-vit-base-patch32"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

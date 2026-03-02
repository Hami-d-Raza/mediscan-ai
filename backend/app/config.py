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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

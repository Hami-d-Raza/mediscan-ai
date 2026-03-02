"""
utils/file_utils.py
-------------------
Reusable helpers for validating uploaded files.
Raises HTTP 400 / 415 errors with clear messages so the frontend can
display meaningful feedback to the user.
"""

from fastapi import UploadFile, HTTPException, status
from app.config import settings


def validate_image_file(file: UploadFile) -> None:
    """
    Validates that the uploaded file is an accepted image format.
    Raises HTTPException on failure.
    """
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type '{file.content_type}'. "
                f"Allowed types: {settings.ALLOWED_IMAGE_TYPES}"
            ),
        )


def validate_report_file(file: UploadFile) -> None:
    """
    Validates that the uploaded file is an accepted report format.
    Raises HTTPException on failure.
    """
    if file.content_type not in settings.ALLOWED_REPORT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported report type '{file.content_type}'. "
                f"Allowed types: {settings.ALLOWED_REPORT_TYPES}"
            ),
        )


def get_file_extension(filename: str) -> str:
    """Returns the lowercase file extension without the dot."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

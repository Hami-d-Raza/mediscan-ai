"""
utils/response_utils.py
-----------------------
Helpers for building consistent JSON response envelopes across all endpoints.
Using a standard envelope makes it easy for the frontend to handle responses.

Example response shape:
{
    "success": true,
    "data": { ... },
    "error": null
}
"""

from typing import Any, Optional


def success_response(data: Any, message: str = "Success") -> dict:
    """Wraps data in a standard success envelope."""
    return {
        "success": True,
        "message": message,
        "data": data,
        "error": None,
    }


def error_response(error: str, message: str = "An error occurred") -> dict:
    """Wraps an error message in a standard error envelope."""
    return {
        "success": False,
        "message": message,
        "data": None,
        "error": error,
    }

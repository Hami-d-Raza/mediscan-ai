"""
routes/health.py
----------------
Simple health-check endpoints.
Used by load balancers / monitoring tools to confirm the API is alive.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check():
    """Returns API health status."""
    return {"status": "ok", "service": "MediScan AI"}

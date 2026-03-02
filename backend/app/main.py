"""
main.py
-------
Entry point of the MediScan AI FastAPI application.
Initializes the app, registers all routers, and configures middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import health, nlp, vision, auth, image, report, mri

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered diagnostic assistance system using NLP and Computer Vision.",
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router,   prefix="/auth",   tags=["Authentication"])
app.include_router(nlp.router,    prefix="/nlp",    tags=["NLP Analysis"])
app.include_router(vision.router, prefix="/vision", tags=["Computer Vision"])
app.include_router(image.router,                    tags=["Image Analysis"])
app.include_router(mri.router,                      tags=["Brain MRI Analysis"])
app.include_router(report.router,                   tags=["Report Analysis"])


# ---------------------------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    """Heartbeat endpoint to confirm the API is running."""
    return {"message": "MediScan AI API Running"}

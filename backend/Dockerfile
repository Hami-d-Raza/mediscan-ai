# ── MediScan AI — Backend Dockerfile ────────────────────────────────────────
# CPU-only image (no CUDA) keeps final size ~900 MB instead of ~5 GB.
# Build:  docker build -t mediscan-backend .
# Run:    docker run -p 8000:8000 --env-file .env mediscan-backend
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for Pillow + sentencepiece
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Install CPU-only PyTorch first (much smaller than default CUDA build)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. Install remaining dependencies
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# ── 3. Copy application source
COPY app/ ./app/
COPY run.py .

# ── 4a. Copy label map (tracked in git, not a large binary)
COPY models/class_labels.json ./models/class_labels.json

# ── 4b. Download model weights from GitHub Releases (not stored in git)
COPY download_models.py .
RUN python download_models.py

# ── 5. Pre-download HuggingFace NLP model so first request isn't slow
#       and Azure startup timeout (230 s) isn't exceeded.
RUN python -c "\
from transformers import pipeline, logging as hf_logging; \
hf_logging.set_verbosity_error(); \
pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='simple'); \
print('NLP model cached.')"

# ── 6. Create uploads directory
RUN mkdir -p uploads

# ── 7. Expose port and start server
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

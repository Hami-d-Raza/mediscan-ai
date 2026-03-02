"""
download_models.py
------------------
Downloads pre-trained model weights from GitHub Releases during Docker build.
Uses only Python standard library — no curl/wget required.
"""
import urllib.request
import os
import sys

MODELS = {
    "models/mri_classifier.pt": (
        "https://github.com/Hami-d-Raza/mediscan-ai"
        "/releases/download/v1.0.1/mri_classifier.pt"
    ),
    "models/multi_cancer_classifier.pt": (
        "https://github.com/Hami-d-Raza/mediscan-ai"
        "/releases/download/v1.0.1/multi_cancer_classifier.pt"
    ),
}

os.makedirs("models", exist_ok=True)

for dest, url in MODELS.items():
    print(f"Downloading {dest} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  -> {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"ERROR downloading {dest}: {e}", file=sys.stderr)
        sys.exit(1)

print("All models downloaded successfully.")

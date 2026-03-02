"""
download_models.py
------------------
Downloads pre-trained model weights from GitHub Releases during Docker build.
Uses only Python standard library — no curl/wget required.

Requirements for this to succeed:
  1. A public GitHub Release tagged "v1.0.1" must exist on the repository.
  2. The two .pt files listed in MODELS must be attached as release assets.

To create the release (one-time setup):
  gh release create v1.0.1 \
      backend/models/mri_classifier.pt \
      backend/models/multi_cancer_classifier.pt \
      --title "v1.0.1 - Model weights" --notes "Initial model release"
"""
import time
import urllib.request
import urllib.error
import os
import sys

RELEASE_BASE = (
    "https://github.com/Hami-d-Raza/mediscan-ai"
    "/releases/download/v1.0.1"
)

MODELS = {
    "models/mri_classifier.pt": f"{RELEASE_BASE}/mri_classifier.pt",
    "models/multi_cancer_classifier.pt": f"{RELEASE_BASE}/multi_cancer_classifier.pt",
}

# Minimum acceptable file size (bytes). Protects against downloading an
# HTML error page that GitHub returns for missing/private assets.
MIN_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def download_file(url: str, dest: str) -> None:
    """Download *url* to *dest* with retries and basic size validation."""
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Attempt {attempt}/{MAX_RETRIES}: {url}", flush=True)
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (mediscan-docker-build)",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = resp.status
                if status != 200:
                    raise RuntimeError(f"HTTP {status} for {url}")
                data = resp.read()

            size = len(data)
            if size < MIN_SIZE_BYTES:
                raise RuntimeError(
                    f"Downloaded file too small ({size} bytes) — "
                    "likely an HTML error page. "
                    "Please ensure the GitHub Release v1.0.1 exists and "
                    "the model assets are attached."
                )

            with open(dest, "wb") as f:
                f.write(data)

            size_mb = size / (1024 * 1024)
            print(f"  -> saved {dest} ({size_mb:.1f} MB)", flush=True)
            return  # success

        except urllib.error.HTTPError as exc:
            msg = (
                f"HTTP {exc.code} {exc.reason} — "
                "check that the GitHub Release v1.0.1 is public and "
                "the asset is attached."
            )
            print(f"  ERROR: {msg}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}", file=sys.stderr)

        if attempt < MAX_RETRIES:
            print(f"  Retrying in {RETRY_DELAY}s …", flush=True)
            time.sleep(RETRY_DELAY)

    # All attempts exhausted
    print(
        f"\nFATAL: Could not download {dest} after {MAX_RETRIES} attempts.\n"
        "Fix checklist:\n"
        "  1. Create the GitHub release:  gh release create v1.0.1 ...\n"
        "  2. Attach both .pt files as release assets.\n"
        "  3. Make sure the repository (and release) are PUBLIC.\n",
        file=sys.stderr,
    )
    sys.exit(1)


os.makedirs("models", exist_ok=True)

for dest, url in MODELS.items():
    print(f"\nDownloading {dest} …", flush=True)
    download_file(url, dest)

print("\nAll models downloaded successfully.", flush=True)

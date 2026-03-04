"""
services/medical_classifier.py
-------------------------------
Binary Medical Image Classifier for MediScan AI.

Classifies an input image as:
  Class 0  Non-Medical  (nature photos, people, objects, everyday scenes)
  Class 1  Medical      (MRI, CT scan, X-ray, histopathology slide)

TIER 1 — Fine-tuned EfficientNet-B0 binary classifier (PRIMARY):
  Weights loaded from `settings.MEDICAL_CLASSIFIER_PATH`.

TIER 2 — Local CLIP zero-shot classifier via transformers (FALLBACK):
  Uses openai/clip-vit-base-patch32 downloaded once and cached locally.
  No external API — runs entirely on CPU.  First call downloads ~600 MB.

Public API:
  is_medical_image(image_path, img=None) -> tuple[bool, float, str]
  Returns (True, score, hint_msg) — never hard-rejects.
  score    : medical likelihood 0–1.
  hint_msg : non-empty string when score is low.
"""

import os
import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate labels for local CLIP zero-shot classification
# ---------------------------------------------------------------------------
_MEDICAL_LABELS = [
    "brain MRI scan",
    "CT scan",
    "chest X-ray radiograph",
    "histopathology microscopy slide",
    "ultrasound scan",
    "medical imaging scan",
]
_NONMEDICAL_LABELS = [
    "regular photograph",
    "natural landscape",
    "artwork or drawing",
    "food photograph",
    "portrait of a person",
    "screenshot",
    "anime illustration",
    "cartoon",
    "wallpaper",
]
_ALL_CLIP_LABELS = _MEDICAL_LABELS + _NONMEDICAL_LABELS

# ---------------------------------------------------------------------------
# Tier-1 threshold: P(medical) must be >= this to be accepted
# ---------------------------------------------------------------------------
MEDICAL_THRESHOLD: float = 0.60

# ---------------------------------------------------------------------------
# Preprocessing pipeline — standard ImageNet normalisation (Tier-1)
# ---------------------------------------------------------------------------
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_finetuned_model: Optional[nn.Module] = None   # Tier-1 binary model
_use_finetuned:   bool = False                   # True once Tier-1 loaded OK
_clip_pipeline   = None                          # Tier-2 local CLIP pipeline
_clip_loaded:    bool = False                    # True once CLIP loading attempted


# ===========================================================================
# Tier-1 — fine-tuned binary EfficientNet-B0
# ===========================================================================

def _build_binary_model() -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2),   # 0=non-medical, 1=medical
    )
    return model


def _load_finetuned() -> bool:
    """
    Try to load the fine-tuned binary weights.
    Returns True on success, False if weights are missing or incompatible.
    """
    global _finetuned_model, _use_finetuned

    if _use_finetuned and _finetuned_model is not None:
        return True

    weights_path = settings.MEDICAL_CLASSIFIER_PATH
    if not os.path.isfile(weights_path):
        logger.info(
            "medical_classifier.pt not found at '%s' — using local CLIP (Tier-2).",
            weights_path,
        )
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build_binary_model()
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    except Exception as exc:
        logger.warning("Could not load binary weights '%s': %s — using fallback.", weights_path, exc)
        return False

    model.to(device)
    model.eval()
    _finetuned_model = model
    _use_finetuned   = True
    logger.info("Tier-1 binary medical classifier loaded from: %s", weights_path)
    return True


# ===========================================================================
# Tier-2 — Local CLIP zero-shot via transformers
# ===========================================================================

def _get_clip_pipeline():
    """
    Lazy-load the local CLIP zero-shot pipeline.
    Downloads openai/clip-vit-base-patch32 (~600 MB) on first call,
    then cached by HuggingFace.  Subsequent calls use the cache.
    """
    global _clip_pipeline, _clip_loaded

    if _clip_loaded:
        return _clip_pipeline   # may be None if loading previously failed

    try:
        from transformers import pipeline as hf_pipeline, logging as hf_logging
        hf_logging.set_verbosity_error()

        logger.info("Loading local CLIP zero-shot pipeline (openai/clip-vit-base-patch32)...")
        _clip_pipeline = hf_pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=-1,   # CPU
        )
        _clip_loaded = True
        logger.info("Local CLIP pipeline loaded successfully.")

    except Exception as exc:
        logger.warning(
            "Could not load local CLIP pipeline: %s — medical validation disabled (pass-through).",
            exc,
        )
        _clip_pipeline = None
        _clip_loaded   = True   # mark attempted so we don't retry every call

    return _clip_pipeline


def _local_clip_score(
    img: Image.Image,
    image_path: str = "<image>",
) -> tuple[bool, float, str]:
    """
    Run local CLIP zero-shot classification to score medical likelihood.
    Returns (True, medical_score, hint_msg).
    Falls back to neutral pass-through (0.70) on any error.
    """
    pipe = _get_clip_pipeline()

    if pipe is None:
        logger.warning(
            "CLIP pipeline unavailable for '%s' — using pass-through score 0.70.",
            image_path,
        )
        return True, 0.70, ""

    try:
        predictions = pipe(img, candidate_labels=_ALL_CLIP_LABELS)
        # Returns list of {"label": str, "score": float} sorted by score desc

        if not predictions:
            logger.warning("CLIP returned empty predictions for '%s'.", image_path)
            return True, 0.70, ""

        score_map     = {p["label"]: p["score"] for p in predictions}
        medical_score = round(
            sum(score_map.get(lbl, 0.0) for lbl in _MEDICAL_LABELS), 4
        )
        top_label = predictions[0]["label"]

        logger.info(
            "Local CLIP [Tier-2]: '%s' | medical_score=%.3f | top='%s'",
            image_path, medical_score, top_label,
        )

        hint_msg = ""
        if medical_score < 0.40:
            hint_msg = (
                f"Low medical likelihood ({medical_score:.2f}) — "
                f"image most resembles '{top_label}'. "
                f"Please upload a genuine medical scan (MRI, CT, X-ray, histopathology)."
            )

        return True, medical_score, hint_msg

    except Exception as exc:
        logger.warning(
            "Local CLIP error for '%s': %s — using pass-through score 0.70.",
            image_path, exc,
        )
        return True, 0.70, ""


# ===========================================================================
# Public API
# ===========================================================================

def is_medical_image(
    image_path: str,
    img: Optional[Image.Image] = None,
) -> tuple[bool, float, str]:
    """
    Determine whether an image is a medical scan or histopathology slide.

    Tries Tier-1 (fine-tuned binary classifier) first; falls back to
    Tier-2 (local CLIP zero-shot pipeline) when Tier-1 weights are absent.

    Args:
        image_path : Path to the image file (used for logging).
        img        : Pre-loaded PIL Image (optional).

    Returns:
        (True, score, hint_msg)
        - Always returns True — no hard rejection; pipeline makes final decision.
        - score    : medical likelihood 0–1 (Tier-1: P(medical); Tier-2: CLIP score).
        - hint_msg : non-empty when score is low; informational only.

    Raises:
        ValueError : If the image cannot be opened.
    """
    # Load image if not supplied
    if img is None:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    # -------------------------------------------------------------------
    # Tier-1: fine-tuned binary classifier
    # -------------------------------------------------------------------
    if _load_finetuned():
        device = next(_finetuned_model.parameters()).device
        input_tensor = _preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = _finetuned_model(input_tensor)    # (1, 2)
            probs  = torch.softmax(logits, dim=1)      # (1, 2)

        medical_prob = round(probs[0][1].item(), 4)    # class-1 = medical
        is_med       = medical_prob >= MEDICAL_THRESHOLD

        hint_msg = ""
        if not is_med:
            hint_msg = (
                f"Low medical probability: {medical_prob * 100:.1f}% "
                f"(reference threshold: {MEDICAL_THRESHOLD * 100:.0f}%). "
                f"Consider uploading a medical scan (MRI, CT, X-ray, histopathology)."
            )
            logger.info("Low medical score [Tier-1]: %s (P=%.3f)", image_path, medical_prob)
        else:
            logger.debug("Medical [Tier-1]: %s (P=%.3f)", image_path, medical_prob)

        # Never hard reject — pass score through for combined pipeline decision
        return True, medical_prob, hint_msg

    # -------------------------------------------------------------------
    # Tier-2: local CLIP zero-shot pipeline
    # -------------------------------------------------------------------
    return _local_clip_score(img, image_path=image_path)

"""
services/mri_model.py
---------------------
Brain MRI classifier for MediScan AI.
MobileNetV2 fine-tuned on the 4-class Brain Tumor MRI Dataset.

4 classes: Glioma · Meningioma · No Tumor · Pituitary Tumor
"""

import os
import math
import hashlib
import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from app.config import settings
from app.services.medical_classifier import is_medical_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels — fixed 4-class Brain MRI dataset
# ---------------------------------------------------------------------------
MRI_CLASS_LABELS: dict[int, str] = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary Tumor",
}

MRI_CLASS_TO_DESCRIPTION: dict[str, str] = {
    "Glioma":          "A tumour that starts in the glial cells of the brain or spine.",
    "Meningioma":      "A tumour arising from the meninges, usually benign and slow-growing.",
    "No Tumor":        "No abnormal growth detected in the brain MRI scan.",
    "Pituitary Tumor": "A growth in the pituitary gland, usually non-cancerous.",
}

# ImageNet normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------
_model: Optional[nn.Module] = None
_simulation_mode: bool = False


def _build_model() -> nn.Module:
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in m.parameters():
        param.requires_grad = False
    in_features = m.classifier[1].in_features  # 1280
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 4),
    )
    return m


def _get_mri_model() -> nn.Module:
    global _model, _simulation_mode

    if _model is not None:
        return _model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Brain MRI MobileNetV2 model on device: %s", device)

    model = _build_model()
    weights_path = settings.MRI_MODEL_PATH

    if os.path.isfile(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            logger.info("Loaded Brain MRI weights from: %s", weights_path)
            _simulation_mode = False
        except Exception as exc:
            logger.warning("Failed to load Brain MRI weights: %s — simulation mode.", exc)
            _simulation_mode = True
    else:
        logger.warning("Brain MRI weights not found at '%s'. Running in SIMULATION mode.", weights_path)
        _simulation_mode = True

    model.to(device)
    model.eval()
    _model = model
    return _model


def _simulate_mri_prediction(image_path: str) -> tuple[int, float]:
    with open(image_path, "rb") as fh:
        digest = hashlib.md5(fh.read()).hexdigest()
    seed = int(digest[-4:], 16)
    class_index = seed % 4
    confidence  = round(0.70 + (seed % 256) / 256 * 0.25, 4)
    return class_index, confidence


# ---------------------------------------------------------------------------
# Decision thresholds — model output only; NO pixel/colour heuristics
# ---------------------------------------------------------------------------
#
# CONFIDENCE_THRESHOLD:
#   If max(softmax) < 0.75 → prediction flagged as "Uncertain".
#   Balanced: permissive enough for real MRI images, cautious enough to
#   flag low-quality or ambiguous inputs.
#
# TOP2_GAP_THRESHOLD:
#   If top-1 prob − top-2 prob < 0.10 → "Uncertain prediction".
# ---------------------------------------------------------------------------
_MRI_CONFIDENCE_THRESHOLD: float = 0.75
_MRI_TOP2_GAP_THRESHOLD:   float = 0.10
_MRI_MAX_ENTROPY_4:        float = math.log(4)   # ≈ 1.386


def _validate_mri_prediction(
    probs_list: list[float],
    confidence: float,
) -> tuple[bool, str]:
    """
    Validate MRI prediction reliability using model output only.
    No pixel-level or colour rules — all checks are probability-based.

    Checks:
      1. Confidence threshold  : max(softmax) >= 0.75
      2. Top-2 probability gap : top1 − top2 >= 0.10
      3. Normalised entropy    : <= 0.85
    """
    # Check 1 — confidence threshold
    if confidence < _MRI_CONFIDENCE_THRESHOLD:
        return False, (
            f"Unknown / Not a medical image. "
            f"Model confidence is too low ({confidence * 100:.1f}%, "
            f"required >= {_MRI_CONFIDENCE_THRESHOLD * 100:.0f}%). "
            f"Please upload a T1/T2 weighted Brain MRI scan."
        )

    # Check 2 — top-2 probability gap
    sorted_probs = sorted(probs_list, reverse=True)
    top2_gap     = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    if top2_gap < _MRI_TOP2_GAP_THRESHOLD:
        return False, (
            f"Uncertain prediction. "
            f"The model is nearly tied between the top-2 classes "
            f"(gap: {top2_gap * 100:.1f}%). "
            f"Please verify with a specialist."
        )

    # Check 3 — normalised entropy
    entropy      = -sum(p * math.log(p + 1e-12) for p in probs_list)
    norm_entropy = entropy / _MRI_MAX_ENTROPY_4
    if norm_entropy > 0.85:
        return False, (
            f"Uncertain prediction. "
            f"Model probability distribution is highly uncertain "
            f"(normalised entropy: {norm_entropy:.2f}). "
            f"Results are unreliable."
        )

    return True, ""


def predict_brain_mri(image_path: str) -> dict:
    """Full pipeline: preprocess → medical gate → Brain MRI classification."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1 — Load and preprocess
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    input_tensor = _preprocess(img).unsqueeze(0)   # (1, 3, 224, 224)

    # Step 2 — Medical likelihood scoring (never hard-rejects; returns score 0–1)
    _, medical_score, med_hint = is_medical_image(image_path, img=img)

    # Step 3 — Brain MRI classification
    model  = _get_mri_model()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)   # (1, 4)

    # Simulation mode — weights not loaded
    if _simulation_mode:
        class_index, confidence = _simulate_mri_prediction(image_path)
        label = MRI_CLASS_LABELS[class_index]
        logger.info("[SIM] Brain MRI: %s | Confidence: %.2f", label, confidence)
        return {
            "prediction":          label,
            "cancer_type":         "Brain Cancer",
            "confidence":          confidence,
            "class_index":         class_index,
            "simulated":           True,
            "description":         MRI_CLASS_TO_DESCRIPTION.get(label, ""),
            "valid_medical_image": False,
            "warning": (
                "Model weights are not loaded — running in SIMULATION mode. "
                "This result is generated from a file hash and has no diagnostic "
                "value. Do not act on this output."
            ),
        }

    probs_list  = probs[0].cpu().tolist()
    class_index = int(torch.argmax(probs, dim=1).item())
    confidence  = round(probs_list[class_index], 4)
    label       = MRI_CLASS_LABELS[class_index]

    # Step 4 — Validate prediction (used only if Rules A/B/D all pass)
    is_valid, validate_warning = _validate_mri_prediction(probs_list, confidence)
    # Keep med_hint separate — do NOT pre-combine here to avoid duplicate messages

    # ------------------------------------------------------------------
    # Step 5 — Combined decision logic
    #
    #  Rule A: confidence < 0.75  → Uncertain
    #  Rule D: medical_score < 0.65  AND  confidence >= 0.90
    #          Suspicious overconfidence (OOD pattern).
    #  Rule B: medical_score < 0.50  AND  confidence < 0.85  → Likely Non-Medical
    #  Rule C: otherwise  → report prediction
    # ------------------------------------------------------------------
    if confidence < _MRI_CONFIDENCE_THRESHOLD:
        logger.info(
            "Uncertain Brain MRI for '%s': confidence=%.3f < %.2f, medical_score=%.3f",
            image_path, confidence, _MRI_CONFIDENCE_THRESHOLD, medical_score,
        )
        return {
            "prediction":          label,
            "cancer_type":         "Brain Cancer",
            "confidence":          confidence,
            "class_index":         class_index,
            "simulated":           False,
            "description":         "",
            "valid_medical_image": False,
            "warning": (
                f"Uncertain prediction — model confidence is too low "
                f"({confidence * 100:.1f}%, required >= {_MRI_CONFIDENCE_THRESHOLD * 100:.0f}%). "
                f"Please upload a T1/T2 weighted Brain MRI scan."
                + (f" | {med_hint}" if med_hint else "")
            ),
        }

    if medical_score < 0.65 and confidence >= 0.90:
        logger.info(
            "Suspicious overconfidence (Rule D) Brain MRI '%s': medical_score=%.3f, confidence=%.3f",
            image_path, medical_score, confidence,
        )
        return {
            "prediction":          "Non-Medical Image",
            "cancer_type":         "N/A",
            "confidence":          confidence,
            "class_index":         -1,
            "simulated":           False,
            "description":         "",
            "valid_medical_image": False,
            "warning": (
                f"This image is likely not a brain MRI. "
                f"The model is suspiciously over-confident ({confidence * 100:.1f}%) on an image "
                f"with a low medical likelihood score ({medical_score:.2f}). "
                f"Please upload a T1/T2 weighted Brain MRI scan."
                + (f" | {med_hint}" if med_hint else "")
            ),
        }

    if medical_score < 0.50 and confidence < 0.85:
        logger.info(
            "Likely non-medical Brain MRI for '%s': medical_score=%.3f, confidence=%.3f",
            image_path, medical_score, confidence,
        )
        return {
            "prediction":          "Likely Non-Medical Image",
            "cancer_type":         "N/A",
            "confidence":          confidence,
            "class_index":         -1,
            "simulated":           False,
            "description":         "",
            "valid_medical_image": False,
            "warning": (
                f"The image has a low medical likelihood score ({medical_score:.2f}) "
                f"combined with borderline confidence ({confidence * 100:.1f}%). "
                f"Please upload a genuine Brain MRI scan."
                + (f" | {med_hint}" if med_hint else "")
            ),
        }

    # Rule C — valid prediction; combine med_hint + validate_warning cleanly
    combined_warning = validate_warning
    if med_hint:
        combined_warning = f"{med_hint} | {validate_warning}" if validate_warning else med_hint

    logger.info(
        "Brain MRI: %s | Confidence: %.2f | Valid: %s | MedScore: %.3f",
        label, confidence, is_valid, medical_score,
    )

    return {
        "prediction":          label,
        "cancer_type":         "Brain Cancer",
        "confidence":          confidence,
        "class_index":         class_index,
        "simulated":           False,
        "description":         MRI_CLASS_TO_DESCRIPTION.get(label, ""),
        "valid_medical_image": is_valid,
        "warning":             combined_warning,
    }

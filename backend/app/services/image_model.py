"""
services/image_model.py
-----------------------
Multi-cancer image analysis service for MediScan AI.
MobileNetV2 fine-tuned on the Kaggle 26-class Multi-Cancer Dataset.

Public API:
  predict_image(image_path)  -> dict   full pipeline result
  predict_cancer(image_path) -> dict   cancer classifier only
"""

import os
import json
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
# Class labels — loaded dynamically from class_labels.json exported during
# Kaggle training so this file never needs editing when retraining.
# Falls back to a minimal set of brain-only labels if JSON is missing.
# ---------------------------------------------------------------------------
_FALLBACK_LABELS = {
    0: "Brain — Glioma",
    1: "Brain — Meningioma",
    2: "Brain — No Tumor",
    3: "Brain — Pituitary Tumor",
}

def _load_class_labels() -> dict[int, str]:
    labels_path = settings.CV_LABELS_PATH
    if os.path.isfile(labels_path):
        with open(labels_path, "r") as f:
            raw = json.load(f)
        # JSON keys are strings; convert to int
        labels = {int(k): v for k, v in raw.items()}
        logger.info("Loaded %d class labels from %s", len(labels), labels_path)
        return labels
    logger.warning(
        "class_labels.json not found at '%s'. Using fallback brain-only labels.",
        labels_path,
    )
    return _FALLBACK_LABELS

CLASS_LABELS: dict[int, str] = _load_class_labels()
NUM_CLASSES: int = len(CLASS_LABELS)

# ImageNet normalisation constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------

# Standard inference transform
_inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Decision thresholds — all based on model output only; NO pixel heuristics
#
# CONFIDENCE_THRESHOLD:
#   If max(softmax) < 0.75 → prediction flagged as "Uncertain".
#   Balanced: high enough to prompt caution, permissive enough for real
#   medical images that may not reach 90 % confidence.
#
# TOP2_GAP_THRESHOLD:
#   If top-1 prob − top-2 prob < 0.10 → "Uncertain prediction".
#   Allows predictions where the model is confident but not overwhelmingly so.
# ---------------------------------------------------------------------------
_CONFIDENCE_THRESHOLD: float = 0.75   # must be >= this to report a result
_TOP2_GAP_THRESHOLD:   float = 0.10   # top-1 minus top-2 must be >= this


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def _build_model(num_classes: int) -> nn.Module:
    """
    Build MobileNetV2 with classifier head replaced for num_classes output.
    Architecture matches the Kaggle training script exactly.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------
_model: Optional[nn.Module] = None
_simulation_mode: bool = False


def _get_model() -> nn.Module:
    """Lazy-load and cache the model. Runs once on the first prediction."""
    global _model, _simulation_mode

    if _model is not None:
        return _model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Multi-Cancer MobileNetV2 model on device: %s", device)

    model = _build_model(NUM_CLASSES)

    weights_path = settings.CV_MODEL_PATH
    if os.path.isfile(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            logger.info("Loaded multi-cancer weights from: %s", weights_path)
            _simulation_mode = False
        except Exception as exc:
            logger.warning(
                "Failed to load weights from '%s': %s — simulation mode.",
                weights_path, exc,
            )
            _simulation_mode = True
    else:
        logger.warning(
            "Model weights not found at '%s'. Running in SIMULATION mode.",
            weights_path,
        )
        _simulation_mode = True

    model.to(device)
    model.eval()
    _model = model
    return _model


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def _simulate_prediction(image_path: str) -> tuple[int, float]:
    """
    Deterministic simulation: derive a class index from an MD5 of the file
    bytes so the same image always returns the same result.
    """
    with open(image_path, "rb") as fh:
        digest = hashlib.md5(fh.read()).hexdigest()

    seed = int(digest[-4:], 16)   # 0 – 65535
    class_index = seed % NUM_CLASSES
    confidence  = round(0.70 + (seed % 256) / 256 * 0.25, 4)
    return class_index, confidence


# ---------------------------------------------------------------------------
# Helper: map a 26-subclass label back to its main cancer type
# ---------------------------------------------------------------------------
_SUBCLASS_TO_TYPE: dict[str, str] = {
    # class_labels.json keys
    "all_benign": "Leukemia (ALL)",   "all_early":  "Leukemia (ALL)",
    "all_pre":    "Leukemia (ALL)",   "all_pro":    "Leukemia (ALL)",
    "brain_glioma": "Brain Cancer",   "brain_menin": "Brain Cancer",
    "brain_tumor":  "Brain Cancer",
    "breast_benign": "Breast Cancer", "breast_malignant": "Breast Cancer",
    "cervix_dyk": "Cervical Cancer",  "cervix_koc": "Cervical Cancer",
    "cervix_mep": "Cervical Cancer",  "cervix_pab": "Cervical Cancer",
    "cervix_sfi": "Cervical Cancer",
    "kidney_normal": "Kidney Cancer", "kidney_tumor": "Kidney Cancer",
    "colon_aca":  "Lung & Colon Cancer", "colon_bnt": "Lung & Colon Cancer",
    "lung_aca":   "Lung & Colon Cancer", "lung_bnt":  "Lung & Colon Cancer",
    "lung_scc":   "Lung & Colon Cancer",
    "lymph_cll":  "Lymphoma",         "lymph_fl":   "Lymphoma",
    "lymph_mcl":  "Lymphoma",
    "oral_normal": "Oral Cancer",     "oral_scc":   "Oral Cancer",
    # fallback label formats ("Brain — Glioma" style)
    "Brain \u2014 Glioma":        "Brain Cancer",
    "Brain \u2014 Meningioma":    "Brain Cancer",
    "Brain \u2014 No Tumor":      "Brain Cancer",
    "Brain \u2014 Pituitary Tumor": "Brain Cancer",
}

def _get_cancer_type(subclass_label: str) -> str:
    return _SUBCLASS_TO_TYPE.get(subclass_label, "Unknown")


# ===========================================================================
# PUBLIC FUNCTIONS — clean modular API
# ===========================================================================

def preprocess_image(image_path: str) -> tuple[torch.Tensor, Image.Image]:
    """Load and preprocess an image from disk for inference."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc
    batch = _inference_transform(pil_img).unsqueeze(0)
    return batch, pil_img


def validate_prediction(
    probs_list: list[float],
    confidence: float,
) -> tuple[bool, str]:
    """
    Check whether a prediction is reliable — based ONLY on model output.
    No pixel-level or colour heuristics are used.

    Three model-based checks:
      1. Confidence threshold   : max(softmax) < 0.75
             → "Uncertain prediction"
      2. Top-2 probability gap  : top1 − top2 < 0.10
             → "Uncertain prediction" (model nearly tied between classes)
      3. Normalised entropy     : > 0.85
             → model spreads probability across many classes → uncertain

    Args:
        probs_list : List of softmax probabilities, one per class.
        confidence : Pre-computed max(probs_list).

    Returns:
        (is_valid, warning_message)
    """
    # Check 1 — confidence threshold
    if confidence < _CONFIDENCE_THRESHOLD:
        return False, (
            f"Unknown / Not a medical image. "
            f"Model confidence is too low ({confidence * 100:.1f}%, "
            f"required >= {_CONFIDENCE_THRESHOLD * 100:.0f}%). "
            f"The image may not belong to any recognised cancer category."
        )

    # Check 2 — top-2 probability gap
    sorted_probs = sorted(probs_list, reverse=True)
    top2_gap     = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    if top2_gap < _TOP2_GAP_THRESHOLD:
        return False, (
            f"Uncertain prediction. "
            f"The model is nearly tied between the top-2 classes "
            f"(gap: {top2_gap * 100:.1f}%). "
            f"Please verify with a specialist."
        )

    # Check 3 — normalised entropy
    max_entropy  = math.log(len(probs_list)) if len(probs_list) > 1 else 1.0
    entropy      = -sum(p * math.log(p + 1e-12) for p in probs_list)
    norm_entropy = entropy / max_entropy
    if norm_entropy > 0.85:
        return False, (
            f"Uncertain prediction. "
            f"Model probability distribution is highly spread "
            f"(normalised entropy: {norm_entropy:.2f}). "
            f"Results are unreliable."
        )

    return True, ""


def predict_cancer(
    image_path: str,
    img: Optional[Image.Image] = None,
    input_batch: Optional[torch.Tensor] = None,
) -> dict:
    """
    Run the 26-class cancer classification model on a (pre-validated) image.

    Callers should call is_medical_image() before this function.
    For the full pipeline, use predict_image() instead.

    Returns:
        {
            "prediction"      : "breast_malignant",
            "cancer_type"     : "Breast Cancer",
            "confidence"      : 0.87,
            "class_index"     : 7,
            "simulated"       : False,
            "valid_prediction": True,
            "warning"         : "",
        }
    """
    if input_batch is None or img is None:
        input_batch, img = preprocess_image(image_path)

    model  = _get_model()
    device = next(model.parameters()).device
    input_batch = input_batch.to(device)

    with torch.no_grad():
        logits        = model(input_batch)               # (1, NUM_CLASSES)
        probabilities = torch.softmax(logits, dim=1)     # (1, NUM_CLASSES)

    # Simulation mode — weights not loaded; use deterministic hash result
    if _simulation_mode:
        class_index, confidence = _simulate_prediction(image_path)
        label       = CLASS_LABELS[class_index]
        cancer_type = _get_cancer_type(label)
        logger.info("[SIM] Prediction: %s | Confidence: %.2f", label, confidence)
        return {
            "prediction":       label,
            "cancer_type":      cancer_type,
            "confidence":       confidence,
            "class_index":      class_index,
            "simulated":        True,
            "valid_prediction": False,
            "warning": (
                "Model weights are not loaded — running in SIMULATION mode. "
                "This result is generated from a file hash and has no diagnostic "
                "value. Do not act on this output."
            ),
        }

    probs_list  = probabilities[0].cpu().tolist()
    class_index = int(torch.argmax(probabilities, dim=1).item())
    confidence  = round(probs_list[class_index], 4)
    label       = CLASS_LABELS[class_index]
    cancer_type = _get_cancer_type(label)

    # Validate using model output only — NO colour/pixel heuristics
    is_valid, warning = validate_prediction(probs_list, confidence)

    logger.info(
        "Cancer prediction: %s (%s) | Confidence: %.2f | Valid: %s",
        label, cancer_type, confidence, is_valid,
    )

    return {
        "prediction":       label,
        "cancer_type":      cancer_type,
        "confidence":       confidence,
        "class_index":      class_index,
        "simulated":        False,
        "valid_prediction": is_valid,
        "warning":          warning,
    }


def predict_image(image_path: str) -> dict:
    """Full pipeline: preprocess → medical gate → cancer classification."""
    # Step 1 — Preprocess
    input_batch, pil_img = preprocess_image(image_path)

    # Step 2 — Medical likelihood scoring (never hard-rejects; returns score 0–1)
    _, medical_score, med_hint = is_medical_image(image_path, img=pil_img)

    # Step 3 — Run cancer classifier regardless of medical score
    result = predict_cancer(image_path, img=pil_img, input_batch=input_batch)

    confidence       = result["confidence"]
    validate_warning = result.get("warning", "")   # from validate_prediction only

    # ------------------------------------------------------------------
    # Step 4 — Combined decision logic
    #
    #  Rule A: confidence < 0.75
    #    Model not confident enough to report reliably.
    #
    #  Rule D: medical_score < 0.65  AND  confidence >= 0.90
    #    SUSPICIOUS OVERCONFIDENCE — model is extremely confident on an image
    #    with a borderline medical score.  Classic OOD pattern (anime/art/photos
    #    that happen to activate cancer-class features).
    #
    #  Rule B: medical_score < 0.50  AND  confidence < 0.85
    #    Low medical likelihood AND borderline confidence.
    #
    #  Rule C: otherwise → accept the prediction as valid.
    # ------------------------------------------------------------------
    if confidence < _CONFIDENCE_THRESHOLD:
        logger.info(
            "Uncertain multi-cancer prediction for '%s': confidence=%.3f < %.2f, "
            "medical_score=%.3f",
            image_path, confidence, _CONFIDENCE_THRESHOLD, medical_score,
        )
        return {
            "prediction":          result["prediction"],
            "cancer_type":         result["cancer_type"],
            "confidence":          confidence,
            "class_index":         result["class_index"],
            "simulated":           result["simulated"],
            "valid_medical_image": False,
            "warning": (
                f"Uncertain prediction — model confidence is too low "
                f"({confidence * 100:.1f}%, required >= {_CONFIDENCE_THRESHOLD * 100:.0f}%). "
                f"Please verify with a specialist."
                + (f" | {med_hint}" if med_hint else "")
            ),
        }

    if medical_score < 0.65 and confidence >= 0.90:
        logger.info(
            "Suspicious overconfidence (Rule D) for '%s': medical_score=%.3f, confidence=%.3f",
            image_path, medical_score, confidence,
        )
        return {
            "prediction":          "Non-Medical Image",
            "cancer_type":         "N/A",
            "confidence":          confidence,
            "class_index":         -1,
            "simulated":           result["simulated"],
            "valid_medical_image": False,
            "warning": (
                f"This image is likely not a medical scan. "
                f"The model is suspiciously over-confident ({confidence * 100:.1f}%) on an image "
                f"with a low medical likelihood score ({medical_score:.2f}). "
                f"This pattern is typical of non-medical images (photos, artwork, anime). "
                f"Please upload a genuine medical scan (MRI, CT, X-ray, histopathology)."
                + (f" | {med_hint}" if med_hint else "")
            ),
        }

    if medical_score < 0.50 and confidence < 0.85:
        logger.info(
            "Likely non-medical image '%s': medical_score=%.3f, confidence=%.3f",
            image_path, medical_score, confidence,
        )
        return {
            "prediction":          "Likely Non-Medical Image",
            "cancer_type":         "N/A",
            "confidence":          confidence,
            "class_index":         result["class_index"],
            "simulated":           result["simulated"],
            "valid_medical_image": False,
            "warning": (
                f"The image has a low medical likelihood score ({medical_score:.2f}) "
                f"combined with borderline confidence ({confidence * 100:.1f}%). "
                f"Please upload a genuine medical scan."
                + (f" | {med_hint}" if med_hint else "")
            ),
        }

    # Rule C — valid prediction; combine med_hint + validate_warning cleanly
    combined_warning = validate_warning
    if med_hint:
        combined_warning = f"{med_hint} | {validate_warning}" if validate_warning else med_hint
    logger.info(
        "Multi-cancer: %s (%s) | Confidence: %.2f | MedScore: %.3f",
        result["prediction"], result["cancer_type"], confidence, medical_score,
    )
    return {
        "prediction":          result["prediction"],
        "cancer_type":         result["cancer_type"],
        "confidence":          confidence,
        "class_index":         result["class_index"],
        "simulated":           result["simulated"],
        "valid_medical_image": result["valid_prediction"],
        "warning":             combined_warning,
    }

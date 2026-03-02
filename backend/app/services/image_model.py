"""
services/image_model.py
-----------------------
PyTorch-based multi-cancer image analysis service for MediScan AI.

Architecture:
  - MobileNetV2 (pretrained on ImageNet) with final FC layer replaced
    to classify medical images into 26 cancer subclasses from the
    Kaggle Multi Cancer Dataset:
    https://www.kaggle.com/datasets/obulisainaren/multi-cancer

    8 main types: ALL, Brain Cancer, Breast, Cervical, Kidney,
                  Lung & Colon, Lymphoma, Oral Cancer
    26 subclasses total (class labels loaded dynamically from class_labels.json)

Weight loading strategy:
  - If `settings.CV_MODEL_PATH` exists on disk → loads fine-tuned weights.
  - If weights file is missing → SIMULATION mode: returns a deterministic
    realistic result from a hash of the image so the full pipeline
    (preprocessing → forward pass) is exercised as in production.

Public API:
  predict_image(image_path: str) -> dict
"""

import os
import json
import hashlib
import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from app.config import settings

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

# ImageNet normalisation — standard for torchvision pretrained models
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# Multi-cancer dataset images are RGB colour JPEGs (512×512 originals).
# Resize to 224×224 for MobileNetV2 input.
# ---------------------------------------------------------------------------
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


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
            state_dict = torch.load(weights_path, map_location=device)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_image(image_path: str) -> dict:
    """
    Run multi-cancer inference and return the predicted subclass with confidence.

    Args:
        image_path: Path to the saved image file (JPG or PNG).

    Returns:
        {
            "prediction":   "brain_glioma",
            "cancer_type":  "Brain Cancer",
            "confidence":   0.87,
            "class_index":  5,
            "simulated":    False,
        }

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError:        If the file cannot be opened as an image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Load and preprocess -------------------------------------------------
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    input_tensor = _preprocess(img)           # (3, 224, 224)
    input_batch  = input_tensor.unsqueeze(0)  # (1, 3, 224, 224)

    # 2. Load model (lazy, cached) -------------------------------------------
    model  = _get_model()
    device = next(model.parameters()).device
    input_batch = input_batch.to(device)

    # 3. Forward pass --------------------------------------------------------
    with torch.no_grad():
        logits = model(input_batch)   # (1, NUM_CLASSES)

        if _simulation_mode:
            class_index, confidence = _simulate_prediction(image_path)
            label = CLASS_LABELS[class_index]
            cancer_type = _get_cancer_type(label)
            logger.info("[SIM] Prediction: %s | Confidence: %.2f", label, confidence)
            return {
                "prediction":  label,
                "cancer_type": cancer_type,
                "confidence":  confidence,
                "class_index": class_index,
                "simulated":   True,
            }

        probabilities = torch.softmax(logits, dim=1)   # (1, NUM_CLASSES)

    # 4. Extract result ------------------------------------------------------
    class_index = int(torch.argmax(probabilities, dim=1).item())
    confidence  = round(probabilities[0][class_index].item(), 4)
    label       = CLASS_LABELS[class_index]
    cancer_type = _get_cancer_type(label)

    logger.info("Prediction: %s (%s) | Confidence: %.2f", label, cancer_type, confidence)

    return {
        "prediction":  label,
        "cancer_type": cancer_type,
        "confidence":  confidence,
        "class_index": class_index,
        "simulated":   False,
    }

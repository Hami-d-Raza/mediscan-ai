"""
services/mri_model.py
---------------------
PyTorch-based Brain MRI classifier for MediScan AI.

Architecture:
  - MobileNetV2 fine-tuned on the Kaggle Brain Tumor MRI Dataset
    (masoudnickparvar/brain-tumor-mri-dataset)

4 classes (alphabetical ImageFolder order):
  0 → glioma       → "Glioma"
  1 → meningioma   → "Meningioma"
  2 → notumor      → "No Tumor"
  3 → pituitary    → "Pituitary Tumor"
"""

import os
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
            state_dict = torch.load(weights_path, map_location=device)
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
# Public API
# ---------------------------------------------------------------------------
def predict_brain_mri(image_path: str) -> dict:
    """
    Run Brain MRI inference.

    Returns:
        {
            "prediction":   "Glioma",
            "cancer_type":  "Brain Cancer",
            "confidence":   0.92,
            "class_index":  0,
            "simulated":    False,
            "description":  "A tumour that starts in the glial cells..."
        }
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    input_tensor = _preprocess(img).unsqueeze(0)

    model  = _get_mri_model()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)

        if _simulation_mode:
            class_index, confidence = _simulate_mri_prediction(image_path)
            label = MRI_CLASS_LABELS[class_index]
            logger.info("[SIM] Brain MRI: %s | Confidence: %.2f", label, confidence)
            return {
                "prediction":  label,
                "cancer_type": "Brain Cancer",
                "confidence":  confidence,
                "class_index": class_index,
                "simulated":   True,
                "description": MRI_CLASS_TO_DESCRIPTION.get(label, ""),
            }

        probs = torch.softmax(logits, dim=1)

    class_index = int(torch.argmax(probs, dim=1).item())
    confidence  = round(probs[0][class_index].item(), 4)
    label       = MRI_CLASS_LABELS[class_index]

    logger.info("Brain MRI Prediction: %s | Confidence: %.2f", label, confidence)

    return {
        "prediction":  label,
        "cancer_type": "Brain Cancer",
        "confidence":  confidence,
        "class_index": class_index,
        "simulated":   False,
        "description": MRI_CLASS_TO_DESCRIPTION.get(label, ""),
    }

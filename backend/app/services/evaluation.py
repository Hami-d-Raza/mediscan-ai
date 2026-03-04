"""
services/evaluation.py
----------------------
Model evaluation utilities for MediScan AI.

Provides proper ML evaluation metrics using scikit-learn:
  - Accuracy
  - Precision (macro / per-class)
  - Recall    (macro / per-class)
  - F1-score  (macro / per-class)
  - Confusion matrix

Public API:
  compute_metrics(y_true, y_pred, class_names) -> dict
  evaluate_model(model, dataloader, device, class_names) -> dict
  print_report(metrics) -> None
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Core metric computation — sklearn-based
# ===========================================================================

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute classification metrics from ground-truth and predicted labels.

    Args:
        y_true       : Ground-truth class indices.
        y_pred       : Predicted class indices.
        class_names  : Human-readable class names (optional; used in report).

    Returns:
        {
            "accuracy"          : 0.93,
            "precision_macro"   : 0.91,
            "recall_macro"      : 0.90,
            "f1_macro"          : 0.905,
            "precision_per_class": [...],   # one float per class
            "recall_per_class"  : [...],
            "f1_per_class"      : [...],
            "confusion_matrix"  : [[...]],  # 2-D list of ints
            "classification_report": "...", # sklearn text report
            "num_samples"       : 1000,
        }
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    accuracy        = float(accuracy_score(y_true_arr, y_pred_arr))
    precision_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    recall_macro    = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_macro        = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    precision_per = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0).tolist()
    recall_per    = recall_score(y_true_arr, y_pred_arr, average=None, zero_division=0).tolist()
    f1_per        = f1_score(y_true_arr, y_pred_arr, average=None, zero_division=0).tolist()
    cm            = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy":               round(accuracy, 4),
        "precision_macro":        round(precision_macro, 4),
        "recall_macro":           round(recall_macro, 4),
        "f1_macro":               round(f1_macro, 4),
        "precision_per_class":    [round(v, 4) for v in precision_per],
        "recall_per_class":       [round(v, 4) for v in recall_per],
        "f1_per_class":           [round(v, 4) for v in f1_per],
        "confusion_matrix":       cm,
        "classification_report":  report,
        "num_samples":            len(y_true),
    }


# ===========================================================================
# Full model evaluation — runs inference on a DataLoader
# ===========================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[list[str]] = None,
    confidence_threshold: float = 0.70,
) -> dict:
    """
    Run the model on all batches in `dataloader` and compute evaluation metrics.

    Args:
        model                : Trained PyTorch model (in eval mode).
        dataloader           : DataLoader yielding (inputs, labels) pairs.
        device               : Torch device (cpu / cuda).
        class_names          : Human-readable class names.
        confidence_threshold : Predictions below this confidence are counted
                               separately to measure out-of-distribution rate.

    Returns:
        dict from compute_metrics() plus:
            "low_confidence_rate" : fraction of predictions below threshold
    """
    model.eval()
    all_true:  list[int] = []
    all_pred:  list[int] = []
    low_conf_count: int  = 0
    total:          int  = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)                         # (B, C)
            probs  = torch.softmax(logits, dim=1)          # (B, C)

            confidences, preds = torch.max(probs, dim=1)   # (B,)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

            low_conf_count += int((confidences < confidence_threshold).sum().item())
            total           += inputs.size(0)

    metrics = compute_metrics(all_true, all_pred, class_names=class_names)
    metrics["low_confidence_rate"] = round(low_conf_count / max(total, 1), 4)
    metrics["confidence_threshold"] = confidence_threshold

    logger.info(
        "Evaluation complete | Accuracy: %.4f | F1: %.4f | Low-conf rate: %.4f",
        metrics["accuracy"], metrics["f1_macro"], metrics["low_confidence_rate"],
    )
    return metrics


# ===========================================================================
# Pretty-print helper
# ===========================================================================

def print_report(metrics: dict, class_names: Optional[list[str]] = None) -> None:
    """
    Print a concise evaluation summary to stdout.

    Args:
        metrics     : dict returned by compute_metrics() or evaluate_model().
        class_names : Class names to display per-class metrics (optional).
    """
    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples        : {metrics.get('num_samples', 'N/A')}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (mac): {metrics['precision_macro']:.4f}")
    print(f"  Recall    (mac): {metrics['recall_macro']:.4f}")
    print(f"  F1-score  (mac): {metrics['f1_macro']:.4f}")

    if "low_confidence_rate" in metrics:
        print(
            f"  Low-conf rate  : {metrics['low_confidence_rate']:.4f}  "
            f"(threshold={metrics.get('confidence_threshold', 0.70):.2f})"
        )

    print("\n  Per-Class Metrics:")
    print(f"  {'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 64)
    for i, (p, r, f) in enumerate(zip(
        metrics["precision_per_class"],
        metrics["recall_per_class"],
        metrics["f1_per_class"],
    )):
        name = class_names[i] if (class_names and i < len(class_names)) else str(i)
        print(f"  {name:<30} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    print("\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    for row in cm:
        print("  " + "  ".join(f"{v:4d}" for v in row))

    print("\n  Classification Report:")
    print(metrics["classification_report"])
    print("=" * 60 + "\n")

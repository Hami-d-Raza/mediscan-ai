"""
train_mri.py
------------
Training script for the MediScan AI Brain Tumor MRI Classifier.

Dataset
-------
Kaggle Brain Tumor MRI Dataset (masoudnickparvar):
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected folder layout after unzipping:
  <dataset_root>/
    Training/
      glioma/       *.jpg
      meningioma/   *.jpg
      notumor/      *.jpg
      pituitary/    *.jpg
    Testing/
      glioma/
      meningioma/
      notumor/
      pituitary/

The four class indices assigned by ImageFolder (alphabetical):
  0 → glioma       → "Glioma"
  1 → meningioma   → "Meningioma"
  2 → notumor      → "Normal (No Tumor)"
  3 → pituitary    → "Pituitary Tumor"

Strategy
--------
Two-phase fine-tuning for fastest convergence + best accuracy on CPU:
  Phase 1 (head-only, default 3 epochs):
    Freeze all MobileNetV2 backbone layers.
    Train only the new classifier head with a higher learning rate.
  Phase 2 (full fine-tune, default remaining epochs):
    Unfreeze all layers and train end-to-end with a lower learning rate.

Model: MobileNetV2 (default) — optimised for CPU, 3-4× faster than ResNet18
       with comparable accuracy on this dataset.

Usage
-----
  python train_mri.py --data <path_to_dataset_root>

  # Full example with custom settings:
  python train_mri.py \
      --data  D:/datasets/brain_tumor_mri \
      --output models/mri_classifier.pt \
      --epochs 20 \
      --head-epochs 5 \
      --batch  32 \
      --lr     1e-3 \
      --finetune-lr 1e-4

Requirements (install separately from the server venv):
  pip install torch torchvision tqdm scikit-learn matplotlib
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

# Use all available CPU cores for PyTorch ops — critical for CPU-only training
_cpu_cores = os.cpu_count() or 4
torch.set_num_threads(_cpu_cores)
torch.set_num_interop_threads(max(1, _cpu_cores // 2))

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False
    print("[warn] tqdm not installed — plain progress printing will be used.")

try:
    from sklearn.metrics import classification_report, confusion_matrix
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    print("[warn] scikit-learn not installed — detailed metrics will be skipped.")

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False
    print("[warn] matplotlib not installed — training plots will be skipped.")


# ---------------------------------------------------------------------------
# Constants (must match app/services/image_model.py)
# ---------------------------------------------------------------------------
CLASS_LABELS = {
    0: "Glioma",
    1: "Meningioma",
    2: "Normal (No Tumor)",
    3: "Pituitary Tumor",
}
NUM_CLASSES = len(CLASS_LABELS)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def get_transforms():
    """
    Returns separate transformations for training and validation/test sets.

    Training augmentation helps the model generalise to:
      - Different MRI orientations (flip / rotation)
      - Brightness / contrast variation across scanners
      - Slightly different zoom levels (random crop)
    """
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Grayscale(num_output_channels=3),   # MRI → 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_datasets(data_root: str):
    """
    Load Training/ and Testing/ folders from the Kaggle dataset root.
    Returns (train_dataset, test_dataset).
    Raises FileNotFoundError if the expected folders are missing.
    """
    train_dir = os.path.join(data_root, "Training")
    test_dir  = os.path.join(data_root, "Testing")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training folder not found at: {train_dir}\n"
            "Please unzip the Kaggle dataset so the 'Training/' folder exists "
            "inside the provided --data directory."
        )
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Testing folder not found at: {test_dir}\n"
            "Please unzip the Kaggle dataset so the 'Testing/' folder exists "
            "inside the provided --data directory."
        )

    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=val_tf)

    # Validate that class folders match our expected labels
    detected = train_ds.classes          # e.g. ['glioma','meningioma','notumor','pituitary']
    expected = ["glioma", "meningioma", "notumor", "pituitary"]
    if detected != expected:
        print(
            f"[warn] Detected class folders {detected} differ from expected {expected}.\n"
            "       Make sure your dataset root contains the unmodified Kaggle folders."
        )

    print(f"\nDataset loaded from: {data_root}")
    print(f"  Training samples : {len(train_ds)}")
    print(f"  Test samples     : {len(test_ds)}")
    print(f"  Class mapping    : {train_ds.class_to_idx}")

    return train_ds, test_ds


def make_weighted_sampler(dataset):
    """
    Build a WeightedRandomSampler so every mini-batch contains an equal
    representation of all classes, compensating for the class imbalance
    present in the Kaggle dataset (~50% pituitary, ~25% glioma, etc.).
    """
    labels  = [label for _, label in dataset.samples]
    counts  = [labels.count(i) for i in range(NUM_CLASSES)]
    weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Model builder (identical architecture to image_model.py)
# ---------------------------------------------------------------------------

def build_model(freeze_backbone: bool = True) -> nn.Module:
    """
    Build MobileNetV2 with classifier head replaced for NUM_CLASSES output.
    MobileNetV2 is 3-4x faster than ResNet18 on CPU with similar accuracy.
    freeze_backbone=True  → only the classifier head is trainable (Phase 1).
    freeze_backbone=False → all layers are trainable (Phase 2).
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = not freeze_backbone
    # Replace the classifier: [Dropout, Linear(1280, 1000)] → Linear(1280, NUM_CLASSES)
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def _bar(iterable, desc="", total=None):
    """Wrap iterable with tqdm if available, else return bare iterable."""
    if _TQDM:
        return tqdm(iterable, desc=desc, total=total, leave=False, ncols=90)
    return iterable


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in _bar(loader, desc="  train"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a data loader. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in _bar(loader, desc="  eval "):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds  .extend(preds.cpu().tolist())
        all_labels .extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_plots(history: dict, out_dir: str):
    """Save loss and accuracy curves as PNG files."""
    if not _MATPLOTLIB:
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Training curves saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if str(device) == "cpu":
        print("[note] Training on CPU. Expect ~30-60 min for 20 epochs on the "
              "full Kaggle dataset. A GPU will be 10-20× faster.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, test_ds = load_datasets(args.data)

    sampler    = make_weighted_sampler(train_ds) if args.weighted_sampler else None
    shuffle_tr = sampler is None  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=shuffle_tr,
        num_workers=args.workers,
        pin_memory=(str(device) == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(str(device) == "cuda"),
    )

    # ------------------------------------------------------------------
    # Phase 1 — train head only (frozen backbone)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Phase 1: Head-only training ({args.head_epochs} epochs)")
    print(f"{'='*60}")

    model     = build_model(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only optimise the classifier head parameters (model.classifier for MobileNetV2)
    head_params = list(model.classifier.parameters())
    optimizer   = torch.optim.Adam(head_params, lr=args.lr)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.head_epochs
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_acc    = 0.0
    best_state  = None

    for epoch in range(1, args.head_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"] .append(tr_acc)
        history["val_loss"]  .append(vl_loss)
        history["val_acc"]   .append(vl_acc)

        elapsed = time.time() - t0
        print(
            f"  [P1] Epoch {epoch:>2}/{args.head_epochs} | "
            f"loss {tr_loss:.4f}/{vl_loss:.4f} | "
            f"acc {tr_acc*100:.1f}%/{vl_acc*100:.1f}% | "
            f"{elapsed:.0f}s"
        )

        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------
    # Phase 2 — full fine-tune (all layers unfrozen)
    # ------------------------------------------------------------------
    remaining = args.epochs - args.head_epochs
    if remaining > 0:
        print(f"\n{'='*60}")
        print(f"Phase 2: Full fine-tune ({remaining} epochs)")
        print(f"{'='*60}")

        # Unfreeze all backbone layers
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining
        )

        for epoch in range(1, remaining + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"] .append(tr_acc)
            history["val_loss"]  .append(vl_loss)
            history["val_acc"]   .append(vl_acc)

            elapsed = time.time() - t0
            print(
                f"  [P2] Epoch {epoch:>2}/{remaining} | "
                f"loss {tr_loss:.4f}/{vl_loss:.4f} | "
                f"acc {tr_acc*100:.1f}%/{vl_acc*100:.1f}% | "
                f"{elapsed:.0f}s"
            )

            if vl_acc > best_acc:
                best_acc   = vl_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  ✓ New best val accuracy: {best_acc*100:.2f}%")

    # ------------------------------------------------------------------
    # Final evaluation on test set with best weights
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Final evaluation on Test set (best weights)")
    print(f"{'='*60}")

    model.load_state_dict(best_state)
    _, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
    print(f"Best Val Acc  : {best_acc * 100:.2f}%")

    if _SKLEARN:
        label_names = [CLASS_LABELS[i] for i in range(NUM_CLASSES)]
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=label_names))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    # ------------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------------
    torch.save(best_state, str(output_path))
    print(f"\nModel weights saved → {output_path}")

    # Save training history to JSON alongside the weights
    history_path = output_path.with_suffix(".json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved → {history_path}")

    # Save plots
    save_plots(history, str(output_path.parent))
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train ResNet18 MRI brain tumor classifier on the Kaggle dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", required=True,
        help="Path to the Kaggle dataset root (contains Training/ and Testing/).",
    )
    p.add_argument(
        "--output", default="models/mri_classifier.pt",
        help="Where to save the best model weights.",
    )
    p.add_argument(
        "--epochs", type=int, default=20,
        help="Total number of training epochs (head + fine-tune).",
    )
    p.add_argument(
        "--head-epochs", type=int, default=5, dest="head_epochs",
        help="Epochs for Phase 1 (head-only, frozen backbone).",
    )
    p.add_argument(
        "--batch", type=int, default=32,
        help="Mini-batch size.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for Phase 1 (head only).",
    )
    p.add_argument(
        "--finetune-lr", type=float, default=1e-4, dest="finetune_lr",
        help="Learning rate for Phase 2 (full fine-tune).",
    )
    p.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader worker processes (0 = main thread, safe on Windows).",
    )
    p.add_argument(
        "--no-weighted-sampler", action="store_false", dest="weighted_sampler",
        help="Disable WeightedRandomSampler (use plain shuffle instead).",
    )
    p.set_defaults(weighted_sampler=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

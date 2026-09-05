"""engine.py

Training / evaluation loops and metric computation (protocol section 14).

Single-logit BCEWithLogitsLoss setup. AUROC (primary endpoint) and secondary
metrics are computed with scikit-learn for determinism. Gradient accumulation
is supported so Swin-T can keep effective batch size = 32 on 8 GB VRAM.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, log_loss,
)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))   # nats


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    """Full metric set (14.1-14.3). Robust to single-class batches (smoke)."""
    labels = labels.astype(int)
    preds = (probs >= threshold).astype(int)
    out: dict[str, Any] = {}

    both_classes = len(np.unique(labels)) == 2
    out["auroc"] = float(roc_auc_score(labels, probs)) if both_classes else float("nan")
    out["auprc"] = float(average_precision_score(labels, probs)) if both_classes else float("nan")

    tn, fp, fn, tp = _safe_confusion(labels, preds)
    out["tp"], out["fp"], out["fn"], out["tn"] = int(tp), int(fp), int(fn), int(tn)
    out["sensitivity"] = _ratio(tp, tp + fn)          # recall / TPR
    out["specificity"] = _ratio(tn, tn + fp)
    out["recall"] = out["sensitivity"]
    out["precision"] = _ratio(tp, tp + fp)            # PPV
    out["ppv"] = out["precision"]
    out["npv"] = _ratio(tn, tn + fn)
    out["accuracy"] = _ratio(tp + tn, tp + tn + fp + fn)
    out["balanced_accuracy"] = (
        (out["sensitivity"] + out["specificity"]) / 2
        if not (math.isnan(out["sensitivity"]) or math.isnan(out["specificity"]))
        else float("nan")
    )
    prec, rec = out["precision"], out["recall"]
    out["f1"] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Calibration baseline (14.3)
    out["brier"] = float(brier_score_loss(labels, probs)) if both_classes else float(np.mean((probs - labels) ** 2))
    try:
        out["nll"] = float(log_loss(labels, probs, labels=[0, 1]))
    except Exception:
        out["nll"] = float("nan")
    out["ece"] = _expected_calibration_error(labels, probs)
    return out


def _safe_confusion(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def _ratio(num, den):
    return float(num) / float(den) if den > 0 else float("nan")


def _expected_calibration_error(labels, probs, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = np.mean(labels[mask])
        conf = np.mean(probs[mask])
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_max_norm,
                    accum_steps: int = 1) -> float:
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels, _meta) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip_max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    # flush a trailing partial accumulation window
    if len(loader) % accum_steps != 0:
        if grad_clip_max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float,
             collect_predictions: bool = False):
    """Returns (metrics, avg_loss, predictions). predictions is [] unless requested."""
    model.eval()
    all_logits, all_labels, all_meta = [], [], []
    total_loss, n = 0.0, 0
    for images, labels, meta in loader:
        images = images.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels_d)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        if collect_predictions:
            all_meta.extend(meta)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    metrics = compute_metrics(labels, probs, threshold)
    avg_loss = total_loss / max(n, 1)

    predictions = []
    if collect_predictions:
        ent = _binary_entropy(probs)
        for i, m in enumerate(all_meta):
            predictions.append({
                "image_id": m["image_id"],
                "relative_path": m["relative_path"],
                "patient_id": m["patient_id"],
                "label": 1 if m["label"] == "PNEUMONIA" else 0,
                "logit": float(logits[i]),
                "probability": float(probs[i]),
                "predicted_label_0_5": int(probs[i] >= threshold),
                "entropy": float(ent[i]),
            })
    return metrics, avg_loss, predictions

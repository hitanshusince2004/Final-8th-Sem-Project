"""
Glacier Melting Detection — Evaluation Metrics
===============================================
All metrics for ML and DL models:

ML Regression:   MAE, MSE, RMSE, R²
ML Classification: Accuracy, Precision, Recall, F1, AUC-ROC
DL Segmentation: IoU (per-class & mean), Dice, Accuracy,
                 Precision, Recall, F1, AUC-ROC, Confusion Matrix
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)


# ══════════════════════════════════════════════════════════════════════════════
# ML REGRESSION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred):
    """
    Compute MAE, MSE, RMSE, R² for regression models.

    Args:
        y_true: array-like of ground truth values
        y_pred: array-like of predicted values

    Returns:
        dict with keys: MAE, MSE, RMSE, R2
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    return {
        "MAE":  round(mae,  4),
        "MSE":  round(mse,  4),
        "RMSE": round(rmse, 4),
        "R2":   round(r2,   4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ML CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def classification_metrics(y_true, y_pred, y_prob=None, average="binary", n_classes=2):
    """
    Compute accuracy, precision, recall, F1, and AUC-ROC.

    Args:
        y_true:    ground truth labels
        y_pred:    predicted class labels
        y_prob:    predicted probabilities (for AUC-ROC)
        average:   'binary', 'macro', or 'weighted'
        n_classes: number of classes

    Returns:
        dict with all classification metrics
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=average, zero_division=0)

    auc = None
    if y_prob is not None:
        try:
            if n_classes == 2:
                prob = np.array(y_prob)
                if prob.ndim == 2:
                    prob = prob[:, 1]
                auc = roc_auc_score(y_true, prob)
            else:
                auc = roc_auc_score(y_true, np.array(y_prob),
                                    multi_class="ovr", average="macro")
        except Exception:
            auc = None

    return {
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
        "AUC_ROC":   round(auc, 4) if auc is not None else "N/A",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION METRICS (PyTorch — used during training)
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    """
    Accumulator for segmentation metrics across batches.
    Supports both binary (num_classes=2) and multi-class modes.

    Usage:
        metrics = SegmentationMetrics(num_classes=4)
        for batch in loader:
            logits, labels = model(batch)
            metrics.update(logits, labels)
        result = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes=4, ignore_index=None):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        self.all_probs  = []
        self.all_labels = []

    def update(self, logits, targets):
        """
        logits:  (B, C, H, W) or (B, C) tensor
        targets: (B, H, W) or (B,)    long tensor
        """
        if logits.ndim == 4: # (B, C, H, W)
            preds  = logits.argmax(dim=1)    # (B, H, W)
            probs  = F.softmax(logits, dim=1)  # (B, C, H, W)
            B, H, W = targets.shape
            
            # Update confusion matrix
            for b in range(B):
                pred_flat   = preds[b].cpu().view(-1)
                target_flat = targets[b].cpu().view(-1)

                if self.ignore_index is not None:
                    valid = target_flat != self.ignore_index
                    pred_flat   = pred_flat[valid]
                    target_flat = target_flat[valid]

                for t, p in zip(target_flat, pred_flat):
                    self.confusion[t.item(), p.item()] += 1

            # Store for AUC computation (subsample to avoid OOM)
            step = max(1, B * H * W // 5000)   # keep ~5k pixels per batch
            probs_np  = probs.detach().cpu().numpy()
            labels_np = targets.detach().cpu().numpy()
            self.all_probs.append(probs_np.transpose(0, 2, 3, 1).reshape(-1, self.num_classes)[::step])
            self.all_labels.append(labels_np.reshape(-1)[::step])
            
        else: # (B, C) classification
            preds = logits.argmax(dim=1) # (B,)
            probs = F.softmax(logits, dim=1) # (B, C)
            B = targets.size(0)
            
            for b in range(B):
                t, p = targets[b].item(), preds[b].item()
                self.confusion[t, p] += 1
                
            self.all_probs.append(probs.detach().cpu().numpy())
            self.all_labels.append(targets.detach().cpu().numpy())

    def compute(self):
        """Return dict of all segmentation metrics."""
        C   = self.num_classes
        cm  = self.confusion.float()

        # Per-class IoU
        tp  = cm.diag()
        fn  = cm.sum(1) - tp
        fp  = cm.sum(0) - tp
        tn  = cm.sum() - tp - fn - fp

        iou_per_class   = tp / (tp + fp + fn + 1e-8)
        dice_per_class  = 2 * tp / (2 * tp + fp + fn + 1e-8)
        prec_per_class  = tp / (tp + fp + 1e-8)
        rec_per_class   = tp / (tp + fn + 1e-8)

        miou  = iou_per_class.mean().item()
        mdice = dice_per_class.mean().item()
        acc   = (tp.sum() / (cm.sum() + 1e-8)).item()
        mprec = prec_per_class.mean().item()
        mrec  = rec_per_class.mean().item()
        mf1   = (2 * mprec * mrec) / (mprec + mrec + 1e-8)

        # AUC-ROC
        auc = None
        if self.all_probs:
            try:
                probs_arr  = np.vstack(self.all_probs)
                labels_arr = np.concatenate(self.all_labels)
                if C == 2:
                    auc = roc_auc_score(labels_arr, probs_arr[:, 1])
                else:
                    auc = roc_auc_score(labels_arr, probs_arr,
                                        multi_class="ovr", average="macro",
                                        labels=list(range(C)))
            except Exception:
                auc = None

        result = {
            "Accuracy":      round(acc,   4),
            "Precision":     round(mprec, 4),
            "Recall":        round(mrec,  4),
            "F1":            round(mf1,   4),
            "mIoU":          round(miou,  4),
            "mDice":         round(mdice, 4),
            "AUC_ROC":       round(auc, 4) if auc is not None else "N/A",
            "IoU_per_class": [round(v.item(), 4) for v in iou_per_class],
            "Dice_per_class":[round(v.item(), 4) for v in dice_per_class],
            "confusion_matrix": cm.long().numpy().tolist(),
        }

        return result


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["Land", "Snow/Ice", "Water", "Debris"]

def print_metrics(metrics_dict, model_name="Model", task="segmentation"):
    """Print a formatted metrics table to stdout."""
    print(f"\n{'═'*55}")
    print(f"  {model_name} — Results")
    print(f"{'═'*55}")

    skip = {"IoU_per_class", "Dice_per_class", "confusion_matrix"}

    for k, v in metrics_dict.items():
        if k in skip:
            continue
        print(f"  {k:<18} {v}")

    if "IoU_per_class" in metrics_dict:
        print(f"\n  Per-class IoU:")
        for i, v in enumerate(metrics_dict["IoU_per_class"]):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
            print(f"    {name:<12} {v:.4f}")

    if "Dice_per_class" in metrics_dict:
        print(f"\n  Per-class Dice:")
        for i, v in enumerate(metrics_dict["Dice_per_class"]):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
            print(f"    {name:<12} {v:.4f}")

    print(f"{'═'*55}\n")


def save_metrics(metrics_dict, path):
    """Save metrics dict to JSON."""
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Metrics saved -> {path}")

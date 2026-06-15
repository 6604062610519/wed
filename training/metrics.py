"""
training/metrics.py — Evaluation metrics สำหรับ wildfire binary segmentation

compute_metrics()      : คำนวณ metric ครบชุดจาก prob/pred arrays
find_best_threshold()  : หา optimal threshold บน validation set (maximize F1)
MetricsTracker         : track metrics across batches + epoch summary
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, cohen_kappa_score,
    jaccard_score
)


# ─────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float = 0.5,
                    prefix: str = "") -> Dict[str, float]:
    """
    คำนวณ metrics ครบชุดสำหรับ binary classification/segmentation

    Parameters
    ----------
    y_true    : ground truth (N,) or (N, H, W) — binary [0, 1]
    y_prob    : predicted probability (N,) or (N, H, W) — float [0, 1]
    threshold : decision threshold
    prefix    : prefix สำหรับ key names ("val_", "test_")
    """
    y_true = y_true.ravel().astype(int)
    y_prob = y_prob.ravel().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Per-class stats
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    # Metrics
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)   # = sensitivity
    specificity = tn / (tn + fp + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)   # Jaccard index สำหรับ fire class

    # Sklearn-based (เพื่อความ robust)
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = average_precision_score(y_true, y_prob)
    except ValueError:
        auc_pr = float("nan")

    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except ValueError:
        kappa = float("nan")

    results = {
        f"{prefix}f1":           f1,
        f"{prefix}precision":    precision,
        f"{prefix}recall":       recall,
        f"{prefix}specificity":  specificity,
        f"{prefix}iou":          iou,
        f"{prefix}auc_roc":      auc_roc,
        f"{prefix}auc_pr":       auc_pr,
        f"{prefix}kappa":        kappa,
        f"{prefix}threshold":    threshold,
        f"{prefix}tp":           int(tp),
        f"{prefix}fp":           int(fp),
        f"{prefix}tn":           int(tn),
        f"{prefix}fn":           int(fn),
        f"{prefix}n_fire":       int(n_pos),
        f"{prefix}n_total":      len(y_true),
        f"{prefix}fire_rate_pred": float(y_pred.mean()),
        f"{prefix}fire_rate_true": float(y_true.mean()),
    }
    return results


# ─────────────────────────────────────────────────────
# Threshold optimization
# ─────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        metric: str = "f1",
                        n_thresholds: int = 50) -> float:
    """
    หา threshold ที่ maximize metric บน validation set
    Default: maximize F1-score

    Returns: best threshold (float)
    """
    y_true = y_true.ravel().astype(int)
    y_prob = y_prob.ravel().astype(float)

    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_val, best_thr = -1, 0.5

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() == 0:      # ไม่มี positive prediction → skip
            continue

        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            val = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            val = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "iou":
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            denom = ((y_pred == 1) | (y_true == 1)).sum()
            val = tp / (denom + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if val > best_val:
            best_val, best_thr = val, thr

    return float(best_thr)


# ─────────────────────────────────────────────────────
# Batch-level tracker
# ─────────────────────────────────────────────────────

class MetricsTracker:
    """
    Track losses + probabilities across batches, summarize per epoch
    ใช้ใน training loop เพื่อ compute epoch-level metrics
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.n_batches  = 0
        self.all_probs  = []
        self.all_labels = []

    def update(self, loss: float, probs: np.ndarray, labels: np.ndarray):
        """
        probs  : (N,) or (N, H, W) float [0, 1]
        labels : (N,) or (N, H, W) int/float [0, 1]
        """
        self.total_loss += loss
        self.n_batches  += 1
        self.all_probs.append(probs.ravel())
        self.all_labels.append(labels.ravel())

    def compute(self, threshold: Optional[float] = None,
                prefix: str = "") -> Dict[str, float]:
        """คำนวณ epoch-level metrics"""
        all_probs  = np.concatenate(self.all_probs)
        all_labels = np.concatenate(self.all_labels)

        if threshold is None:
            threshold = find_best_threshold(all_labels, all_probs)

        metrics = compute_metrics(all_labels, all_probs, threshold, prefix)
        metrics[f"{prefix}loss"] = self.total_loss / max(self.n_batches, 1)
        return metrics

    def avg_loss(self) -> float:
        return self.total_loss / max(self.n_batches, 1)


# ─────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────

def print_metrics(metrics: dict, title: str = ""):
    """แสดง metrics เป็น table อ่านง่าย"""
    if title:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")

    key_order = ["loss", "f1", "precision", "recall", "specificity",
                 "iou", "auc_roc", "auc_pr", "kappa", "threshold"]

    for key in key_order:
        # ค้นหา key ที่อาจมี prefix
        for k, v in metrics.items():
            if k.endswith(key) and isinstance(v, float):
                bar = "█" * int(v * 20) if 0 <= v <= 1 else ""
                print(f"  {k:<22s}: {v:7.4f}  {bar}")
                break

    if "n_fire" in metrics or any(k.endswith("n_fire") for k in metrics):
        for k, v in metrics.items():
            if k.endswith("n_fire"):
                n = metrics.get(k.replace("n_fire", "n_total"), 0)
                rate = metrics.get(k.replace("n_fire", "fire_rate_pred"), 0)
                print(f"\n  Fire pixels: {v:,} / {n:,} ({rate*100:.2f}% predicted)")
                break


if __name__ == "__main__":
    print("=== Testing Metrics ===")
    rng = np.random.default_rng(42)
    n = 10000
    y_true = (rng.random(n) < 0.05).astype(float)
    y_prob = rng.beta(0.5, 5, n)
    y_prob[y_true == 1] += rng.beta(1, 2, int(y_true.sum()))
    y_prob = np.clip(y_prob, 0, 1)

    best_thr = find_best_threshold(y_true, y_prob)
    print(f"  Best threshold: {best_thr:.3f}")

    m = compute_metrics(y_true, y_prob, threshold=best_thr, prefix="test_")
    print_metrics(m, "Test Metrics")

    print("\n✅ metrics.py OK")

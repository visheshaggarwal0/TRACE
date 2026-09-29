from typing import Dict
import numpy as np


def compute_pixel_metrics(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes standard pixel-level classification metrics for road extraction."""
    pred_bin = (pred_mask >= threshold).astype(bool)
    true_bin = (true_mask >= 0.5).astype(bool)

    tp = np.logical_and(pred_bin, true_bin).sum()
    fp = np.logical_and(pred_bin, ~true_bin).sum()
    fn = np.logical_and(~pred_bin, true_bin).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    return {
        "pixel_iou": iou,
        "pixel_f1": f1,
        "pixel_precision": precision,
        "pixel_recall": recall,
    }

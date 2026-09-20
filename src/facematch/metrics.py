"""Verification metrics from genuine/impostor distance samples (pure numpy).

Convention: a pair is *accepted* when distance <= threshold.
  FAR (false accept rate)  = fraction of impostor pairs accepted
  FRR (false reject rate)  = fraction of genuine pairs rejected
  TAR (true accept rate)   = 1 - FRR
"""

from __future__ import annotations

import numpy as np


def roc_curve(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, far, tar), one point per distinct observed distance."""
    genuine, impostor = np.sort(np.asarray(genuine, float)), np.sort(np.asarray(impostor, float))
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Need at least one genuine and one impostor distance")
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    tar = np.searchsorted(genuine, thresholds, side="right") / genuine.size
    far = np.searchsorted(impostor, thresholds, side="right") / impostor.size
    return thresholds, far, tar


def auc(far: np.ndarray, tar: np.ndarray) -> float:
    x = np.concatenate([[0.0], far, [1.0]])
    y = np.concatenate([[0.0], tar, [1.0]])
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def equal_error_rate(thresholds: np.ndarray, far: np.ndarray, tar: np.ndarray) -> tuple[float, float]:
    """Return (eer, threshold_at_eer): the point where FAR == FRR."""
    frr = 1.0 - tar
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thresholds[i])


def tar_at_far(far: np.ndarray, tar: np.ndarray, target_far: float) -> float:
    ok = far <= target_far
    return float(tar[ok].max()) if ok.any() else 0.0


def best_accuracy_threshold(
    genuine: np.ndarray, impostor: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """Threshold maximising balanced accuracy, and that accuracy."""
    _, far, tar = roc_curve(genuine, impostor)
    bal_acc = (tar + (1.0 - far)) / 2
    i = int(np.argmax(bal_acc))
    return float(thresholds[i]), float(bal_acc[i])


def summarize(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    thresholds, far, tar = roc_curve(genuine, impostor)
    eer, eer_thr = equal_error_rate(thresholds, far, tar)
    best_thr, bal_acc = best_accuracy_threshold(genuine, impostor, thresholds)
    return {
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "auc": round(auc(far, tar), 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_thr, 4),
        "best_threshold": round(best_thr, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "tar_at_far_1e-2": round(tar_at_far(far, tar, 1e-2), 4),
        "tar_at_far_1e-3": round(tar_at_far(far, tar, 1e-3), 4),
    }

import numpy as np
import pytest

from facematch.metrics import roc_curve, summarize


def test_perfect_separation():
    s = summarize(np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9]))
    assert s["auc"] == 1.0
    assert s["eer"] == 0.0
    assert s["balanced_accuracy"] == 1.0
    assert 0.3 <= s["best_threshold"] < 0.7


def test_fully_overlapping_is_chance():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 1, 5000)
    s = summarize(d[:2500], d[2500:])
    assert s["auc"] == pytest.approx(0.5, abs=0.03)
    assert s["eer"] == pytest.approx(0.5, abs=0.03)


def test_inverted_scores_give_auc_zero():
    assert summarize(np.array([0.8, 0.9]), np.array([0.1, 0.2]))["auc"] == 0.0


def test_roc_is_monotonic():
    rng = np.random.default_rng(1)
    _, far, tar = roc_curve(rng.normal(0.3, 0.1, 300), rng.normal(0.7, 0.1, 300))
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(tar) >= 0)


def test_requires_both_classes():
    with pytest.raises(ValueError):
        roc_curve(np.array([]), np.array([0.5]))

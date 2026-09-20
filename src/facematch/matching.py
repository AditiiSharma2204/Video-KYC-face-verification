"""Distance and decision logic (pure numpy, fully unit-tested)."""

from __future__ import annotations

import numpy as np

from .models import Decision


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. 0 = identical direction, 1 = orthogonal."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cannot compare zero-norm embeddings")
    return float(1.0 - np.dot(a, b) / denom)


def decide(distance: float, threshold: float, review_margin: float = 0.15) -> Decision:
    """Three-way decision.

    * distance <= threshold*(1-margin)  -> MATCH     (confidently the same person)
    * distance >  threshold*(1+margin)  -> NO_MATCH  (confidently different)
    * in between                        -> REVIEW    (route to a human; avoids
      forcing a coin-flip on borderline pairs, which is what KYC ops teams want)
    """
    if not 0 <= review_margin < 1:
        raise ValueError("review_margin must be in [0, 1)")
    if distance <= threshold * (1 - review_margin):
        return Decision.MATCH
    if distance > threshold * (1 + review_margin):
        return Decision.NO_MATCH
    return Decision.REVIEW


def similarity_score(distance: float, threshold: float) -> float:
    """Map distance to a 0-100 display score where 50 sits exactly on the threshold.

    Presentation only - decisions are made on distance, never on this number.
    """
    return float(np.clip(100.0 * (1.0 - distance / (2.0 * threshold)), 0.0, 100.0))

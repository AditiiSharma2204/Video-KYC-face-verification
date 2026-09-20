"""Plain data containers shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Decision(str, Enum):
    MATCH = "match"
    REVIEW = "review"  # inside the uncertainty band: route to a human
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class FaceBox:
    """A detected face. ``landmarks`` is (5, 2): two eyes, nose, two mouth corners."""

    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: np.ndarray | None = None

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True)
class QualityReport:
    sharpness: float  # variance of Laplacian; higher is sharper
    brightness: float  # mean gray level, 0-255
    face_px: int  # shorter side of the face box in pixels
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


@dataclass
class VerificationResult:
    decision: Decision
    distance: float
    threshold: float
    similarity: float  # 0-100 score derived from distance/threshold, for display only
    model: str
    detector: str
    document_quality: QualityReport
    selfie_quality: QualityReport
    liveness_score: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.decision is Decision.MATCH

    def to_dict(self) -> dict:
        def quality(q: QualityReport) -> dict:
            return {
                "sharpness": round(q.sharpness, 2),
                "brightness": round(q.brightness, 2),
                "face_px": q.face_px,
                "issues": q.issues,
            }

        return {
            "decision": self.decision.value,
            "verified": self.verified,
            "distance": round(self.distance, 4),
            "threshold": round(self.threshold, 4),
            "similarity": round(self.similarity, 1),
            "model": self.model,
            "detector": self.detector,
            "liveness_score": None if self.liveness_score is None else round(self.liveness_score, 3),
            "document_quality": quality(self.document_quality),
            "selfie_quality": quality(self.selfie_quality),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }

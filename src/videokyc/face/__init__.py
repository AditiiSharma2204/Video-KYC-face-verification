"""Face verification: detection, alignment, quality gating, embedding, matching."""

from .config import FaceConfig
from .models import Decision, FaceBox, QualityReport, VerificationResult
from .pipeline import Face, FaceMatchPipeline

__all__ = [
    "Decision", "Face", "FaceBox", "FaceConfig", "FaceMatchPipeline", "QualityReport", "VerificationResult",
]

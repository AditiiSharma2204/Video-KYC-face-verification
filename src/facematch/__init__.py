"""Aadhaar face match: verify a live selfie against the photo on an Aadhaar document."""

from .config import Config
from .errors import FaceMatchError
from .models import Decision, VerificationResult
from .pipeline import FaceMatchPipeline

__all__ = ["Config", "Decision", "FaceMatchError", "FaceMatchPipeline", "VerificationResult"]
__version__ = "1.0.0"

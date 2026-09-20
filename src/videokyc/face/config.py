"""Central configuration. All tunables live here so they can be swept in evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Cosine-distance thresholds shipped with DeepFace (tuned on LFW). Use
# ``python -m videokyc.face.evaluate`` to calibrate on your own data.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "ArcFace": 0.68,
    "Facenet512": 0.30,
    "Facenet": 0.40,
    "VGG-Face": 0.68,
    "SFace": 0.593,
}


@dataclass
class FaceConfig:
    model_name: str = "ArcFace"
    detector: str = "yunet"  # "yunet" (default) or "haar" (no download needed)
    threshold: float | None = None  # None -> DEFAULT_THRESHOLDS[model_name]
    review_margin: float = 0.15  # distances within +/-15% of threshold => REVIEW
    pdf_dpi: int = 300
    face_size: int = 224  # crop side for 'box' alignment (ArcFace alignment is always 112)
    alignment: str = "auto"  # "auto" | "arcface" (5-point template) | "box" (generic eye-levelled crop)
    crop_margin: float = 0.25  # extra context around the detected box
    min_detection_score: float = 0.6
    # quality gates
    min_face_px: int = 60
    min_sharpness: float = 40.0
    brightness_range: tuple[float, float] = (45.0, 220.0)
    enforce_quality: bool = True
    # presentation-attack detection on the selfie (needs DeepFace + torch weights)
    liveness: bool = False
    model_dir: str = field(default_factory=lambda: os.environ.get(
        "FACEMATCH_MODEL_DIR", os.path.join(os.path.expanduser("~"), ".facematch")))

    @property
    def effective_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        try:
            return DEFAULT_THRESHOLDS[self.model_name]
        except KeyError:
            raise ValueError(
                f"No default threshold for model {self.model_name!r}; set FaceConfig.threshold."
            ) from None

    @classmethod
    def from_env(cls) -> FaceConfig:
        """Build a FaceConfig from FACEMATCH_* environment variables (used by the API/Docker)."""
        cfg = cls()
        cfg.model_name = os.environ.get("FACEMATCH_MODEL", cfg.model_name)
        cfg.detector = os.environ.get("FACEMATCH_DETECTOR", cfg.detector)
        if "FACEMATCH_THRESHOLD" in os.environ:
            cfg.threshold = float(os.environ["FACEMATCH_THRESHOLD"])
        cfg.liveness = os.environ.get("FACEMATCH_LIVENESS", "0") == "1"
        return cfg

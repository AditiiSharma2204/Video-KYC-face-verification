"""Cheap, dependency-free image-quality gates run before the (expensive) embedding.

Measured on the face region of the *original* image, resampled to a fixed size
so the sharpness score doesn't depend on how large the face happens to be.
The default thresholds are conservative starting points, not calibrated values.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .models import FaceBox, QualityReport

_PROBE = 128


def assess(img_bgr: np.ndarray, box: FaceBox, config: Config) -> QualityReport:
    h, w = img_bgr.shape[:2]
    x0, y0 = max(box.x, 0), max(box.y, 0)
    region = img_bgr[y0 : min(box.y + box.h, h), x0 : min(box.x + box.w, w)]
    if region.size == 0:
        return QualityReport(0.0, 0.0, 0, ["face region is empty"])

    probe = cv2.resize(region, (_PROBE, _PROBE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    face_px = int(min(box.w, box.h))

    issues: list[str] = []
    if face_px < config.min_face_px:
        issues.append(f"face too small ({face_px}px < {config.min_face_px}px)")
    if sharpness < config.min_sharpness:
        issues.append("image is blurry")
    lo, hi = config.brightness_range
    if brightness < lo:
        issues.append("image is too dark")
    elif brightness > hi:
        issues.append("image is overexposed")
    return QualityReport(sharpness, brightness, face_px, issues)

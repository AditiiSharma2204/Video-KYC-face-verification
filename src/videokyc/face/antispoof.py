"""Passive presentation-attack detection for the live selfie.

Uses DeepFace's bundled Fasnet (MiniFASNet) anti-spoofing model, which flags
printed photos and screen replays. It is a *passive* check - a determined
attacker with a high-quality 3D mask or injection attack can still get past it,
so treat it as one layer, not a guarantee.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..errors import NoFaceError


class LivenessChecker(Protocol):
    def check(self, img_bgr: np.ndarray) -> tuple[bool, float]:
        """Return (is_real, score in [0, 1])."""
        ...


class DeepFaceLiveness:
    def check(self, img_bgr: np.ndarray) -> tuple[bool, float]:
        from deepface import DeepFace

        try:
            faces = DeepFace.extract_faces(
                img_path=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                detector_backend="opencv",
                enforce_detection=True,
                anti_spoofing=True,
            )
        except ValueError as exc:
            raise NoFaceError("live capture") from exc
        face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
        return bool(face["is_real"]), float(face["antispoof_score"])

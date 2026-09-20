"""Face embedding backends.

``Embedder`` is a Protocol so the pipeline (and its tests) never depend on
TensorFlow. ``DeepFaceEmbedder`` imports DeepFace lazily.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class Embedder(Protocol):
    model_name: str

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised 1-D embedding for an already detected+aligned face."""
        ...


class DeepFaceEmbedder:
    def __init__(self, model_name: str = "ArcFace") -> None:
        self.model_name = model_name
        self._warm = False

    def embed(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        from deepface import DeepFace

        # detector_backend="skip" bypasses DeepFace's own detection (we already did it
        # and aligned the face). In that mode DeepFace flips channels once before the
        # model, so we hand it RGB to end up with the BGR order the models expect.
        rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        out = DeepFace.represent(
            img_path=rgb,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        return normalize(np.asarray(out[0]["embedding"], dtype=np.float64))


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-norm embedding")
    return v / n

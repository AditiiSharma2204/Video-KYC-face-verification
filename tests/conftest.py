from __future__ import annotations

import cv2
import numpy as np
import pytest

from facematch import Config, FaceMatchPipeline
from facematch.models import FaceBox


def textured(seed: int = 0, size: int = 240, mean: int = 128) -> np.ndarray:
    """Sharp, well-lit synthetic image that passes the quality gates."""
    rng = np.random.default_rng(seed)
    img = rng.normal(mean, 40, (size, size, 3)).clip(0, 255).astype(np.uint8)
    return img


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class FakeDetector:
    name = "fake"

    def __init__(self, boxes: list[FaceBox] | None = None):
        self.boxes = [FaceBox(60, 60, 120, 120, 0.99)] if boxes is None else boxes

    def detect(self, img_bgr):
        return self.boxes


class FakeEmbedder:
    """Returns queued vectors in call order (document first, then selfie)."""

    model_name = "fake"

    def __init__(self, *vectors):
        self.queue = [np.asarray(v, float) / np.linalg.norm(v) for v in vectors]

    def embed(self, face):
        return self.queue.pop(0)


class FakeLiveness:
    def __init__(self, real: bool):
        self.real = real

    def check(self, img):
        return self.real, 0.9 if self.real else 0.1


@pytest.fixture
def make_pipeline():
    def _make(embeddings, detector=None, liveness=None, **cfg):
        config = Config(model_name="ArcFace", **cfg)  # threshold 0.68
        return FaceMatchPipeline(
            config, detector or FakeDetector(), FakeEmbedder(*embeddings), liveness
        )

    return _make

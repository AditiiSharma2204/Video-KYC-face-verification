"""Eye tracking backends that turn a video frame into an EAR value."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from .blink import mean_ear

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
LANDMARKER_FILE = "face_landmarker.task"


class EyeTracker(Protocol):
    def ear(self, frame_bgr: np.ndarray) -> float | None:
        """Mean eye aspect ratio for the most prominent face, or None if no face."""
        ...


def ensure_landmarker(model_dir: str) -> Path:
    path = Path(model_dir) / LANDMARKER_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(LANDMARKER_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(path)
    return path


class MediaPipeEyeTracker:
    """MediaPipe FaceLandmarker (478 landmarks). One instance per video stream/thread."""

    def __init__(self, model_dir: str) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_landmarker(model_dir))),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def ear(self, frame_bgr: np.ndarray) -> float | None:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])
        return mean_ear(pts)

    def close(self) -> None:
        self._landmarker.close()

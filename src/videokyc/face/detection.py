"""Face detection, alignment and cropping.

Two interchangeable detectors:

* ``YuNetDetector``  - OpenCV's YuNet ONNX model. Fast, robust to the small
  passport-style photo on an Aadhaar card, and returns 5 landmarks used for
  eye-level alignment. The 230 KB model is downloaded once and cached.
* ``HaarDetector``   - the classic cascade shipped with OpenCV. Zero download,
  weaker accuracy, no landmarks. Kept as an offline fallback.
"""

from __future__ import annotations

import math
import threading
import urllib.request
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import FaceConfig
from .models import FaceBox

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FILE = "face_detection_yunet_2023mar.onnx"


class FaceDetector(Protocol):
    name: str

    def detect(self, img_bgr: np.ndarray) -> list[FaceBox]: ...


class HaarDetector:
    name = "haar"

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, img_bgr: np.ndarray) -> list[FaceBox]:
        gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        rects = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        # Haar gives no confidence; use 1.0 so the score gate never rejects it.
        return [FaceBox(int(x), int(y), int(w), int(h), 1.0) for x, y, w, h in rects]


class YuNetDetector:
    name = "yunet"

    def __init__(self, model_path: str | Path, score_threshold: float = 0.6) -> None:
        self._model_path = str(model_path)
        self._score_threshold = score_threshold
        self._det = None  # created lazily, then reused: loading the ONNX model per frame is very slow
        self._lock = threading.Lock()  # FaceDetectorYN keeps mutable input-size state

    def detect(self, img_bgr: np.ndarray) -> list[FaceBox]:
        h, w = img_bgr.shape[:2]
        # Cap the long side so a 300-dpi A4 render doesn't slow detection; boxes are rescaled back.
        scale = min(1.0, 1600 / max(h, w))
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale) if scale < 1 else img_bgr
        with self._lock:
            if self._det is None:
                self._det = cv2.FaceDetectorYN.create(
                    self._model_path, "", (small.shape[1], small.shape[0]), self._score_threshold, 0.3, 5000
                )
            self._det.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self._det.detect(small)
        if rows is None:
            return []
        boxes = []
        for r in rows:
            x, y, bw, bh = (r[:4] / scale).tolist()
            lm = (r[4:14].reshape(5, 2) / scale).astype(np.float32)
            boxes.append(FaceBox(int(x), int(y), int(bw), int(bh), float(r[14]), lm))
        return boxes


def ensure_model(config: FaceConfig) -> Path:
    """Return the local YuNet model path, downloading it on first use."""
    path = Path(config.model_dir) / YUNET_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(YUNET_URL, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(path)
    return path


def build_detector(config: FaceConfig) -> FaceDetector:
    if config.detector == "haar":
        return HaarDetector()
    if config.detector == "yunet":
        return YuNetDetector(ensure_model(config), config.min_detection_score)
    raise ValueError(f"Unknown detector {config.detector!r}; choose 'yunet' or 'haar'.")


def best_face(boxes: list[FaceBox]) -> FaceBox | None:
    """Pick the most likely subject: largest area, confidence as tie-breaker.

    (The original prototype took ``faces[0]``, which is arbitrary.)
    """
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b.area, b.score))


def align_crop(img_bgr: np.ndarray, box: FaceBox, size: int = 224, margin: float = 0.25) -> np.ndarray:
    """Return a ``size`` x ``size`` crop with eyes levelled (when landmarks exist).

    Rotation, scaling and cropping are folded into a single affine warp, so
    faces near the image edge are padded rather than truncated.
    """
    side = max(box.w, box.h) * (1 + 2 * margin)
    angle = 0.0
    pivot = box.center
    if box.landmarks is not None:
        left, right = sorted(box.landmarks[:2].tolist(), key=lambda p: p[0])
        angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
        pivot = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)

    m = cv2.getRotationMatrix2D(pivot, angle, size / side)
    cx, cy = box.center
    tx, ty = m @ np.array([cx, cy, 1.0])
    m[:, 2] += (size / 2 - tx, size / 2 - ty)
    return cv2.warpAffine(img_bgr, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# Canonical 5-point template ArcFace/InsightFace models are trained on (112x112 crops):
# [eye (image-left), eye (image-right), nose tip, mouth corner (image-left), mouth corner (image-right)]
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def align_arcface(img_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp the face so its 5 landmarks land on the canonical ArcFace template.

    Recognition embeddings are only as consistent as the alignment feeding them. Using the
    exact geometry the model was trained on (similarity transform: rotation + uniform scale
    + translation, no shear) matters most for small, low-quality ID photos.
    """
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE * (size / 112.0), method=cv2.LMEDS)
    if matrix is None:  # degenerate landmarks
        raise ValueError("Could not estimate an alignment transform from the landmarks")
    return cv2.warpAffine(img_bgr, matrix, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def align_face(img_bgr: np.ndarray, box: FaceBox, config: FaceConfig) -> np.ndarray:
    """Pick the alignment that suits the configured model.

    ``arcface`` uses the 5-point template (needs YuNet landmarks); ``box`` is the generic
    eye-levelled crop around the detected box (works with any detector/model).
    """
    mode = config.alignment
    if mode == "auto":
        mode = "arcface" if config.model_name == "ArcFace" and box.landmarks is not None else "box"
    if mode == "arcface":
        if box.landmarks is None:
            raise ValueError("ArcFace alignment needs facial landmarks; use detector='yunet' or alignment='box'")
        return align_arcface(img_bgr, box.landmarks, 112)
    return align_crop(img_bgr, box, config.face_size, config.crop_margin)

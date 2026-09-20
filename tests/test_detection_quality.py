import cv2
import numpy as np

from facematch import detection, quality
from facematch.config import Config
from facematch.models import FaceBox

from .conftest import textured


def test_best_face_prefers_largest_then_score():
    small = FaceBox(0, 0, 40, 40, 0.99)
    big_low = FaceBox(0, 0, 100, 100, 0.7)
    big_high = FaceBox(0, 0, 100, 100, 0.9)
    assert detection.best_face([small, big_low, big_high]) is big_high
    assert detection.best_face([]) is None


def _eye_levelness(crop: np.ndarray) -> float:
    """|dy| between the two bright eye blobs in the aligned crop."""
    mask = (crop[:, :, 0] > 200).astype(np.uint8)
    n, _, _, cents = cv2.connectedComponentsWithStats(mask)
    assert n == 3, f"expected 2 eye blobs, found {n - 1}"
    (_, y1), (_, y2) = sorted(cents[1:].tolist())
    return abs(y1 - y2)


def test_align_crop_levels_tilted_eyes():
    img = np.zeros((300, 300, 3), np.uint8)
    left, right = (100, 130), (180, 160)  # right eye sits 30px lower => tilted face
    for p in (left, right):
        cv2.circle(img, p, 5, (255, 255, 255), -1)
    landmarks = np.array([left, right, (140, 170), (115, 200), (165, 205)], np.float32)
    box = FaceBox(80, 100, 120, 130, 0.99, landmarks)

    aligned = detection.align_crop(img, box, size=128)
    unaligned = detection.align_crop(img, FaceBox(80, 100, 120, 130, 0.99), size=128)

    assert aligned.shape == (128, 128, 3)
    assert _eye_levelness(aligned) < 1.5
    assert _eye_levelness(unaligned) > 8  # sanity: without landmarks the tilt remains


def test_align_crop_pads_faces_at_image_edge():
    img = textured(size=100)
    crop = detection.align_crop(img, FaceBox(0, 0, 60, 60, 0.9), size=64)
    assert crop.shape == (64, 64, 3)


def test_quality_flags_blur_dark_small():
    cfg = Config()
    box = FaceBox(20, 20, 150, 150, 0.9)
    assert quality.assess(textured(), box, cfg).passed

    blurred = cv2.GaussianBlur(textured(), (0, 0), 6)
    assert "image is blurry" in quality.assess(blurred, box, cfg).issues

    dark = (textured() * 0.1).astype(np.uint8)
    assert "image is too dark" in quality.assess(dark, box, cfg).issues

    tiny = quality.assess(textured(), FaceBox(20, 20, 30, 30, 0.9), cfg)
    assert any("too small" in i for i in tiny.issues)


def test_quality_handles_box_outside_image():
    report = quality.assess(textured(size=50), FaceBox(500, 500, 40, 40, 0.9), Config())
    assert not report.passed

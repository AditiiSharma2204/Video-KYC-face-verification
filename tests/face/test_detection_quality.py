import cv2
import numpy as np
import pytest

from tests.conftest import textured
from videokyc.face import detection, quality
from videokyc.face.config import FaceConfig
from videokyc.face.models import FaceBox


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


def _place_template(scale, angle_deg, tx, ty):
    """The canonical ArcFace landmarks after a similarity transform (what YuNet would report)."""
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (detection.ARCFACE_TEMPLATE @ rot.T) * scale + np.array([tx, ty])


@pytest.mark.parametrize("scale, angle, tx, ty", [(1.0, 0, 0, 0), (2.5, 0, 90, 40), (1.8, 12, 120, 60), (3.0, -20, 200, 80)])
def test_arcface_alignment_maps_landmarks_onto_the_canonical_template(scale, angle, tx, ty):
    pts = _place_template(scale, angle, tx, ty)
    img = np.zeros((480, 640, 3), np.uint8)
    for p in pts:
        cv2.circle(img, tuple(int(v) for v in p), max(2, int(2 * scale)), (255, 255, 255), -1)

    out = detection.align_arcface(img, pts, 112)
    assert out.shape == (112, 112, 3)
    mask = (out[:, :, 0] > 128).astype(np.uint8)
    n, _, _, cents = cv2.connectedComponentsWithStats(mask)
    assert n - 1 == 5, f"expected 5 landmark blobs, found {n - 1}"
    found = cents[1:]
    for want in detection.ARCFACE_TEMPLATE:  # every template point has a blob within 2px
        assert np.min(np.linalg.norm(found - want, axis=1)) < 2.0


def test_arcface_alignment_rejects_degenerate_landmarks():
    with pytest.raises(ValueError):
        detection.align_arcface(np.zeros((50, 50, 3), np.uint8), np.zeros((5, 2)))


def test_align_face_mode_selection():
    img = textured(size=300)
    with_lm = FaceBox(80, 80, 120, 140, 0.9, _place_template(1.6, 0, 60, 60).astype(np.float32))
    no_lm = FaceBox(80, 80, 120, 140, 0.9)

    assert detection.align_face(img, with_lm, FaceConfig(model_name="ArcFace")).shape == (112, 112, 3)  # auto -> arcface
    assert detection.align_face(img, no_lm, FaceConfig(model_name="ArcFace")).shape == (224, 224, 3)  # auto -> box
    assert detection.align_face(img, with_lm, FaceConfig(model_name="Facenet512", threshold=0.3)).shape == (224, 224, 3)
    assert detection.align_face(img, with_lm, FaceConfig(alignment="box")).shape == (224, 224, 3)
    with pytest.raises(ValueError, match="landmarks"):
        detection.align_face(img, no_lm, FaceConfig(alignment="arcface"))


def test_align_crop_pads_faces_at_image_edge():
    img = textured(size=100)
    crop = detection.align_crop(img, FaceBox(0, 0, 60, 60, 0.9), size=64)
    assert crop.shape == (64, 64, 3)


def test_quality_flags_blur_dark_small():
    cfg = FaceConfig()
    box = FaceBox(20, 20, 150, 150, 0.9)
    assert quality.assess(textured(), box, cfg).passed

    blurred = cv2.GaussianBlur(textured(), (0, 0), 6)
    assert "image is blurry" in quality.assess(blurred, box, cfg).issues

    dark = (textured() * 0.1).astype(np.uint8)
    assert "image is too dark" in quality.assess(dark, box, cfg).issues

    tiny = quality.assess(textured(), FaceBox(20, 20, 30, 30, 0.9), cfg)
    assert any("too small" in i for i in tiny.issues)


def test_quality_handles_box_outside_image():
    report = quality.assess(textured(size=50), FaceBox(500, 500, 40, 40, 0.9), FaceConfig())
    assert not report.passed

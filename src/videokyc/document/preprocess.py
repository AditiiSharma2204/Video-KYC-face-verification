"""Image clean-up before OCR: upscale, contrast, denoise, deskew, binarise.

ID cards are photographed on tables, scanned crooked and printed on coloured
security backgrounds, so no single preprocessing recipe wins. ``variants``
returns a few candidates and the OCR layer keeps whichever reads best.
"""

from __future__ import annotations

import cv2
import numpy as np

TARGET_WIDTH = 2000  # Tesseract works best around 300 dpi; a 2000px-wide card is ~ that.
MAX_DESKEW_DEG = 15.0


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def upscale(gray: np.ndarray, target_width: int = TARGET_WIDTH) -> np.ndarray:
    h, w = gray.shape[:2]
    if w >= target_width:
        return gray
    scale = target_width / w
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees (positive = text rotated counter-clockwise), 0 if unsure."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(bw)
    if len(xs) < 500:
        return 0.0
    # Merge characters into text lines so minAreaRect follows the lines, not glyphs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 60), 3))
    merged = cv2.dilate(bw, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles, weights = [], []
    for c in contours:
        (_, _), (w, h), ang = cv2.minAreaRect(c)
        if max(w, h) < gray.shape[1] * 0.08 or min(w, h) == 0 or max(w, h) / min(w, h) < 3:
            continue  # ignore blobs that are not text-line shaped
        if w < h:
            ang += 90
        ang = ((ang + 45) % 90) - 45  # normalise to [-45, 45)
        angles.append(ang)
        weights.append(max(w, h))
    if not angles:
        return 0.0
    skew = float(np.average(angles, weights=weights))
    return skew if 0.3 <= abs(skew) <= MAX_DESKEW_DEG else 0.0


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def variants(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Return candidate images for OCR, cheapest/most reliable first."""
    gray = upscale(to_gray(img_bgr))
    gray = rotate(gray, estimate_skew(gray))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.medianBlur(clahe, 3)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [clahe, otsu]

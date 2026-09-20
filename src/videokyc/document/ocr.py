"""OCR engines. ``OcrEngine`` is a Protocol so the rest of the pipeline is testable
without Tesseract installed."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..errors import OcrUnavailableError
from . import preprocess
from .ocr_text import lines_from_data

_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float  # mean word confidence, 0-100 (-1 if unknown)


class OcrEngine(Protocol):
    def read(self, img_bgr: np.ndarray) -> OcrResult: ...


def find_tesseract() -> str | None:
    """Locate the tesseract binary: $TESSERACT_CMD, then PATH, then the Windows default."""
    for candidate in (os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), _WINDOWS_DEFAULT):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


class TesseractOcr:
    """Tesseract via pytesseract, run on several preprocessed variants; best-confidence wins.

    Words below ``min_word_conf`` are dropped: junk read from photos, holograms and card
    borders next to the text is almost always low-confidence.
    """

    def __init__(self, lang: str = "eng", psm: int = 6, min_word_conf: int = 40) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - depends on install
            raise OcrUnavailableError("pytesseract is not installed (pip install pytesseract).") from exc
        cmd = find_tesseract()
        if cmd is None:
            raise OcrUnavailableError(
                "Tesseract binary not found. Install it (Windows: winget install UB-Mannheim.TesseractOCR; "
                "Debian/Ubuntu: apt install tesseract-ocr) or set TESSERACT_CMD."
            )
        pytesseract.pytesseract.tesseract_cmd = cmd
        self._pt = pytesseract
        self._config = f"--oem 3 --psm {psm}"
        self._lang = lang
        self._min_conf = min_word_conf

    def read(self, img_bgr: np.ndarray) -> OcrResult:
        best, best_mass = OcrResult("", -1.0), -1.0
        for variant in preprocess.variants(img_bgr):
            data = self._pt.image_to_data(
                variant, lang=self._lang, config=self._config, output_type=self._pt.Output.DICT
            )
            text, conf, n_words = lines_from_data(data, self._min_conf)
            # Judge variants by total confidence mass (words read x confidence), not the mean:
            # a variant that confidently reads three words must not beat one that reads thirty.
            mass = conf * n_words
            if mass > best_mass:
                best, best_mass = OcrResult(text, conf), mass
        return best

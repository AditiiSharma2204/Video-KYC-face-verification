"""Turn uploaded bytes (PDF or image) into BGR numpy arrays, entirely in memory.

PDFs are rendered with PyMuPDF, which needs no external poppler install and
supports password-protected files (UIDAI e-Aadhaar PDFs are encrypted).
"""

from __future__ import annotations

import cv2
import numpy as np

from .errors import DocumentError, PasswordRequiredError, WrongPasswordError

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 50_000_000  # guards against decompression bombs

_PDF_MAGIC = b"%PDF"


def is_pdf(data: bytes) -> bool:
    return data[:5].lstrip(b"\x00\xef\xbb\xbf ").startswith(_PDF_MAGIC)


def load_image(data: bytes) -> np.ndarray:
    """Decode an image file to a BGR uint8 array (EXIF orientation is applied)."""
    _check_size(data)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise DocumentError("Could not decode image; expected JPEG or PNG.")
    if arr.shape[0] * arr.shape[1] > MAX_PIXELS:
        raise DocumentError("Image dimensions are too large.")
    return arr


def render_pdf(data: bytes, password: str | None = None, dpi: int = 300, page: int = 0) -> np.ndarray:
    """Render one PDF page to a BGR array."""
    import pymupdf  # imported lazily so the rest of the package works without it

    _check_size(data)
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # pymupdf raises several distinct types
        raise DocumentError(f"Could not open PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            if not password:
                raise PasswordRequiredError("This PDF is password protected.")
            if not doc.authenticate(password):
                raise WrongPasswordError("Incorrect PDF password.")
        if doc.page_count == 0:
            raise DocumentError("PDF has no pages.")
        pix = doc[min(page, doc.page_count - 1)].get_pixmap(dpi=dpi, alpha=False)
        if pix.width * pix.height > MAX_PIXELS:
            raise DocumentError("PDF page is too large to render.")
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)


def load_document(data: bytes, password: str | None = None, dpi: int = 300) -> np.ndarray:
    """Load an Aadhaar document, which may be a PDF or a scanned/photographed image."""
    return render_pdf(data, password, dpi) if is_pdf(data) else load_image(data)


def _check_size(data: bytes) -> None:
    if not data:
        raise DocumentError("Uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise DocumentError(f"File exceeds the {MAX_BYTES // (1024 * 1024)} MB limit.")

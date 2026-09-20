"""Document stage: validate -> OCR -> classify -> extract fields -> extract ID photo.

    upload --> intake (magic bytes, size, password, pages)
           --> OCR (preprocess + Tesseract) --> classify --> extract fields
           --> ID photograph (largest face across pages)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import NoFaceError
from ..face.pipeline import Face, FaceMatchPipeline
from . import intake
from .classify import Classification, classify
from .extract import IdFields, extract_fields
from .ocr import OcrEngine


@dataclass
class DocumentResult:
    classification: Classification
    fields: IdFields
    photo: Face
    ocr_confidence: float
    n_pages: int


class DocumentProcessor:
    def __init__(self, ocr: OcrEngine, face: FaceMatchPipeline, dpi: int = 300, max_pages: int = 3) -> None:
        self.ocr = ocr
        self.face = face
        self.dpi = dpi
        self.max_pages = max_pages

    def process(self, data: bytes, password: str | None = None) -> DocumentResult:
        pages = intake.load_pages(data, password, self.dpi, self.max_pages)

        texts, confs = [], []
        for page in pages:
            res = self.ocr.read(page)
            texts.append(res.text)
            if res.confidence >= 0:
                confs.append(res.confidence)
        text = "\n".join(texts)

        classification = classify(text)
        fields = extract_fields(text, classification.doc_type)
        photo = self._best_photo(pages)
        return DocumentResult(
            classification, fields, photo, float(np.mean(confs)) if confs else -1.0, len(pages)
        )

    def _best_photo(self, pages: list[np.ndarray]) -> Face:
        """The ID photograph is the largest confidently-detected face across all pages."""
        best: Face | None = None
        for page in pages:
            try:
                face = self.face.find_face(page, "ID document")
            except NoFaceError:
                continue
            if best is None or face.box.area > best.box.area:
                best = face
        if best is None:
            raise NoFaceError("ID document")
        return best

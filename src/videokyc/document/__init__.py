"""Government-ID intake: validation, OCR, classification, field and photo extraction."""

from .classify import Classification, DocType, classify
from .extract import IdFields, extract_fields
from .ocr import OcrEngine, OcrResult, TesseractOcr
from .processor import DocumentProcessor, DocumentResult

__all__ = [
    "Classification", "DocType", "DocumentProcessor", "DocumentResult", "IdFields",
    "OcrEngine", "OcrResult", "TesseractOcr", "classify", "extract_fields",
]

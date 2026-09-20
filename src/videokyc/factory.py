"""Build a production KycEngine from environment variables.

    FACEMATCH_MODEL / FACEMATCH_DETECTOR / FACEMATCH_THRESHOLD   face matching (see FaceConfig.from_env)
    KYC_STT            whisper (default) | none    speech-to-text for audio answers
    KYC_WHISPER_MODEL  base.en (default)           any faster-whisper model name
    KYC_OCR_LANG       eng (default)               Tesseract language(s), e.g. eng+hin
    MONGO_URI          if set, outcomes go to MongoDB (else in-memory)
    MONGO_DB / MONGO_TTL_DAYS                      database name / record retention
    KYC_HASH_KEY       secret for keyed ID hashing (privacy.hash_id)
"""

from __future__ import annotations

import os

from .document import DocumentProcessor, TesseractOcr
from .face import FaceConfig, FaceMatchPipeline
from .liveness.landmarks import MediaPipeEyeTracker
from .session.engine import KycConfig, KycEngine
from .storage.base import KycStore
from .storage.memory import MemoryStore


def build_store() -> KycStore:
    uri = os.environ.get("MONGO_URI")
    if not uri:
        return MemoryStore()
    from .storage.mongo import MongoStore

    ttl = os.environ.get("MONGO_TTL_DAYS")
    return MongoStore(uri, db=os.environ.get("MONGO_DB", "videokyc"), ttl_days=int(ttl) if ttl else None)


def build_engine(config: KycConfig | None = None) -> KycEngine:
    if os.environ.get("MONGO_URI") and not os.environ.get("KYC_HASH_KEY"):
        # The ID hash is only privacy-preserving if its key is secret; the built-in default is public.
        raise RuntimeError("KYC_HASH_KEY must be set when MONGO_URI is configured (use a long random secret).")
    face_cfg = FaceConfig.from_env()
    face = FaceMatchPipeline(face_cfg)
    ocr = TesseractOcr(lang=os.environ.get("KYC_OCR_LANG", "eng"))

    stt = None
    if os.environ.get("KYC_STT", "whisper") != "none":
        from .voice.stt import FasterWhisperStt

        stt = FasterWhisperStt(os.environ.get("KYC_WHISPER_MODEL", "base.en"))

    return KycEngine(
        documents=DocumentProcessor(ocr, face, dpi=face_cfg.pdf_dpi),
        face=face,
        eyes_factory=lambda: MediaPipeEyeTracker(face_cfg.model_dir),
        store=build_store(),
        stt=stt,
        config=config,
    )

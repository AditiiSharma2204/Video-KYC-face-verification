"""End-to-end verification: document + selfie -> decision.

    bytes -> ingest -> detect -> quality gate -> align/crop -> embed -+
                                                                      +-> distance -> decision
    bytes -> ingest -> detect -> quality gate -> align/crop -> embed -+
                                     (+ optional liveness on selfie)

Nothing is written to disk; images live only for the duration of the call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import detection, ingest, quality
from .config import Config
from .detection import FaceDetector
from .embedding import Embedder
from .errors import LowQualityError, NoFaceError, SpoofDetectedError
from .liveness import LivenessChecker
from .matching import cosine_distance, decide, similarity_score
from .models import FaceBox, QualityReport, VerificationResult


@dataclass
class Face:
    """A detected, aligned face plus the evidence behind it (useful for UIs)."""

    crop: np.ndarray  # aligned BGR crop
    box: FaceBox
    quality: QualityReport


class FaceMatchPipeline:
    def __init__(
        self,
        config: Config | None = None,
        detector: FaceDetector | None = None,
        embedder: Embedder | None = None,
        liveness: LivenessChecker | None = None,
    ) -> None:
        self.config = config or Config()
        self.detector = detector or detection.build_detector(self.config)
        if embedder is None:
            from .embedding import DeepFaceEmbedder

            embedder = DeepFaceEmbedder(self.config.model_name)
        self.embedder = embedder
        if liveness is None and self.config.liveness:
            from .liveness import DeepFaceLiveness

            liveness = DeepFaceLiveness()
        self.liveness = liveness

    # -- stages -----------------------------------------------------------------

    def find_face(self, img_bgr: np.ndarray, source: str) -> Face:
        box = detection.best_face(self.detector.detect(img_bgr))
        if box is None:
            raise NoFaceError(source)
        crop = detection.align_crop(img_bgr, box, self.config.face_size, self.config.crop_margin)
        return Face(crop, box, quality.assess(img_bgr, box, self.config))

    # -- public API -------------------------------------------------------------

    def verify(self, document: bytes, selfie: bytes, password: str | None = None) -> VerificationResult:
        t0 = time.perf_counter()
        doc_img = ingest.load_document(document, password, self.config.pdf_dpi)
        selfie_img = ingest.load_image(selfie)
        t_ingest = time.perf_counter()
        return self.verify_images(doc_img, selfie_img, _timings={"ingest": (t_ingest - t0) * 1000})

    def verify_images(
        self, doc_img: np.ndarray, selfie_img: np.ndarray, _timings: dict[str, float] | None = None
    ) -> VerificationResult:
        timings = _timings or {}
        t = time.perf_counter()
        doc_face = self.find_face(doc_img, "Aadhaar document")
        selfie_face = self.find_face(selfie_img, "live capture")
        timings["detect"] = (time.perf_counter() - t) * 1000
        return self.compare(doc_face, selfie_face, selfie_img, timings)

    def compare(
        self,
        doc_face: Face,
        selfie_face: Face,
        selfie_img: np.ndarray,
        timings: dict[str, float] | None = None,
    ) -> VerificationResult:
        """Quality gate, optional liveness, embed and decide, given faces already located."""
        timings = timings if timings is not None else {}
        cfg = self.config

        # Only the live capture is hard-gated: the Aadhaar photo is small and low-res by
        # design, so its quality is reported as a warning instead of rejected.
        if cfg.enforce_quality and not selfie_face.quality.passed:
            raise LowQualityError("Live capture", selfie_face.quality.issues)

        liveness_score = None
        if self.liveness is not None:
            t = time.perf_counter()
            is_real, liveness_score = self.liveness.check(selfie_img)
            timings["liveness"] = (time.perf_counter() - t) * 1000
            if not is_real:
                raise SpoofDetectedError("Live capture appears to be a photo or screen replay.")

        t = time.perf_counter()
        distance = cosine_distance(self.embedder.embed(doc_face.crop), self.embedder.embed(selfie_face.crop))
        timings["embed_and_match"] = (time.perf_counter() - t) * 1000

        threshold = cfg.effective_threshold
        return VerificationResult(
            decision=decide(distance, threshold, cfg.review_margin),
            distance=distance,
            threshold=threshold,
            similarity=similarity_score(distance, threshold),
            model=cfg.model_name,
            detector=self.detector.name,
            document_quality=doc_face.quality,
            selfie_quality=selfie_face.quality,
            liveness_score=liveness_score,
            timings_ms=timings,
        )

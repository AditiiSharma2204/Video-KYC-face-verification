"""The Video-KYC orchestrator. Implements the full flow:

    upload ID -> validate -> OCR -> classify -> extract fields + ID photo
      -> live webcam: face detection + blink liveness + quality gate
      -> face verification (ArcFace, median over the best live frames)
           NO  -> reject
           YES -> voice KYC (10 questions, answers checked against the ID)
      -> outcome (approved / manual review / rejected) persisted to the store

Recoverable problems (wrong file, no blink yet, unclear answer) raise or return so the
UI can retry; terminal decisions move the session to REJECTED/COMPLETED and persist it.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from ..document.classify import DocType
from ..document.processor import DocumentProcessor
from ..errors import (
    DocumentError,
    LivenessFailedError,
    NoFaceError,
    PasswordRequiredError,
    SessionNotFoundError,
    SessionStateError,
    SpoofDetectedError,
    UnsupportedDocumentError,
    WrongPasswordError,
)
from ..face.models import Decision
from ..face.pipeline import FaceMatchPipeline
from ..liveness.blink import BlinkConfig
from ..liveness.capture import LiveCapture, run_video
from ..liveness.landmarks import EyeTracker
from ..privacy import hash_id, mask_id
from ..storage.base import KycStore
from ..voice.flow import VoiceFlow
from ..voice.questions import AnswerResult, Question, Verdict, VoiceConfig
from ..voice.stt import SpeechToText
from .models import KycSession, Outcome, Stage


@dataclass
class KycConfig:
    allowed_doc_types: frozenset[DocType] = frozenset(t for t in DocType if t is not DocType.UNKNOWN)
    min_doc_confidence: float = 0.2
    max_document_attempts: int = 3
    max_video_attempts: int = 3
    review_action: Literal["continue", "reject"] = "continue"  # what to do on a borderline face match
    session_ttl_s: int = 1800
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)


@dataclass
class AnswerOutcome:
    result: AnswerResult
    retry: bool  # same question will be asked again
    next_question: Question | None
    session: KycSession


class KycEngine:
    def __init__(
        self,
        documents: DocumentProcessor,
        face: FaceMatchPipeline,
        eyes_factory: Callable[[], EyeTracker],
        store: KycStore,
        stt: SpeechToText | None = None,
        config: KycConfig | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.documents = documents
        self.face = face
        self.eyes_factory = eyes_factory
        self.store = store
        self.stt = stt
        self.config = config or KycConfig()
        self._clock = clock
        self._sessions: dict[str, KycSession] = {}
        self._doc_attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------

    def create_session(self) -> KycSession:
        s = KycSession(id=uuid.uuid4().hex, created_at=self._clock())
        s.log(self._clock(), "session_created")
        with self._lock:
            self._sessions[s.id] = s
        self._persist(s)
        return s

    def get(self, session_id: str) -> KycSession:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise SessionNotFoundError(f"No session {session_id!r}")
        age = (self._clock() - s.created_at).total_seconds()
        if not s.terminal and age > self.config.session_ttl_s:
            self._reject(s, "Session expired")
        return s

    def _require(self, session_id: str, stage: Stage) -> KycSession:
        s = self.get(session_id)
        if s.stage is not stage:
            raise SessionStateError(f"Session is in stage '{s.stage.value}', expected '{stage.value}'.")
        return s

    def _persist(self, s: KycSession) -> None:
        self.store.save(s.to_record())

    def _reject(self, s: KycSession, reason: str) -> None:
        s.rejection_reason = reason
        s.stage, s.outcome = Stage.REJECTED, Outcome.REJECTED
        s.log(self._clock(), "rejected", reason=reason)
        self._release_runtime(s)
        self._persist(s)

    def _release_runtime(self, s: KycSession) -> None:
        """Drop PII/images from memory as soon as the session ends."""
        s.fields = s.doc_photo = s.voice_flow = None

    # -- stage 1: document -----------------------------------------------------------

    def submit_document(self, session_id: str, data: bytes, password: str | None = None) -> KycSession:
        s = self._require(session_id, Stage.DOCUMENT)
        try:
            doc = self.documents.process(data, password)
            cfg = self.config
            if doc.classification.doc_type not in cfg.allowed_doc_types or (
                doc.classification.confidence < cfg.min_doc_confidence
            ):
                raise UnsupportedDocumentError(
                    "This does not look like a supported ID (Aadhaar, PAN, Passport, Voter ID, Driving Licence)."
                )
        except PasswordRequiredError:
            raise  # not an attempt: the user just needs to supply the password
        except (DocumentError, NoFaceError, UnsupportedDocumentError, WrongPasswordError) as exc:
            n = self._doc_attempts.get(s.id, 0) + 1
            self._doc_attempts[s.id] = n
            s.log(self._clock(), "document_rejected", reason=type(exc).__name__, attempt=n)
            if n >= self.config.max_document_attempts:
                self._reject(s, "Too many invalid document uploads")
            raise

        f = doc.fields
        s.doc_type = doc.classification.doc_type.value
        s.doc_confidence = doc.classification.confidence
        s.ocr_confidence = round(doc.ocr_confidence, 1)
        if f.id_number:
            s.id_masked, s.id_hash = mask_id(f.id_number), hash_id(f.id_number)
            s.duplicate_id_seen = self.store.count_approved_by_hash(s.id_hash) > 0
        s.doc_warnings = list(f.warnings)
        s.fields, s.doc_photo = f, doc.photo
        s.stage = Stage.LIVENESS
        s.log(
            self._clock(), "document_accepted", doc_type=s.doc_type, confidence=s.doc_confidence,
            warnings=f.warnings, duplicate_id=s.duplicate_id_seen,
        )
        self._persist(s)
        return s

    # -- stage 2: live capture, liveness, face match -----------------------------------------

    def new_capture(self, session_id: str) -> LiveCapture:
        """A fresh per-frame processor for streaming UIs (WebRTC). Feed it frames, then call
        ``complete_liveness``."""
        self._require(session_id, Stage.LIVENESS)
        return LiveCapture(self.face, self.eyes_factory(), self.config.blink)

    def submit_video(self, session_id: str, data: bytes) -> KycSession:
        """Batch alternative to streaming: decode an uploaded video and run the same checks."""
        capture = run_video(self.new_capture(session_id), data)
        return self.complete_liveness(session_id, capture)

    def complete_liveness(self, session_id: str, capture: LiveCapture) -> KycSession:
        s = self._require(session_id, Stage.LIVENESS)
        s.video_attempts += 1
        left = self.config.max_video_attempts - s.video_attempts

        def fail(reason: str) -> LivenessFailedError:
            s.log(self._clock(), "liveness_failed", reason=reason, attempts_left=max(left, 0))
            if left <= 0:
                self._reject(s, f"Liveness check failed: {reason}")
            else:
                self._persist(s)
            return LivenessFailedError(reason, max(left, 0))

        s.blinks = capture.blink.blinks
        if not capture.blink_passed:
            raise fail(f"blink not detected ({capture.blink.blinks}/{self.config.blink.required_blinks} blinks)")
        candidates = capture.best_frames()
        if not candidates:
            raise fail("no clear, well-lit face frames captured")
        s.log(self._clock(), "liveness_passed", blinks=s.blinks, frames=len(candidates))

        best = candidates[0]
        try:
            result = self.face.compare(
                s.doc_photo, best.face, best.frame, extra_faces=tuple(c.face for c in candidates[1:])
            )
        except SpoofDetectedError as exc:
            self._reject(s, str(exc))
            raise

        s.face = result.to_dict()
        s.log(self._clock(), "face_matched", decision=result.decision.value, distance=round(result.distance, 4))

        if result.decision is Decision.NO_MATCH:
            self._reject(s, "Face does not match the ID photograph")
            return s
        if result.decision is Decision.REVIEW:
            if self.config.review_action == "reject":
                self._reject(s, "Face match is borderline")
                return s
            s.needs_review = True

        s.stage = Stage.VOICE
        s.voice_flow = VoiceFlow(s.fields, self.config.voice)
        self._persist(s)
        return s

    # -- stage 3: voice KYC ------------------------------------------------------------------------

    def current_question(self, session_id: str) -> Question | None:
        s = self._require(session_id, Stage.VOICE)
        return s.voice_flow.current

    def submit_answer(
        self, session_id: str, transcript: str | None = None, audio: bytes | None = None
    ) -> AnswerOutcome:
        s = self._require(session_id, Stage.VOICE)
        if transcript is None:
            if audio is None:
                raise ValueError("Provide a transcript or audio")
            if self.stt is None:
                raise SessionStateError("No speech-to-text engine configured; submit a transcript instead.")
            transcript = self.stt.transcribe(audio)

        flow = s.voice_flow
        result = flow.answer(transcript)
        if flow.retry_pending:
            s.log(self._clock(), "answer_retry", question=result.question_id, verdict=result.verdict.value)
            return AnswerOutcome(result, True, flow.current, s)

        s.voice.append(
            {"question": result.question_id, "verdict": result.verdict.value, "score": result.score,
             "detail": result.detail}
        )
        s.log(self._clock(), "answer_recorded", question=result.question_id, verdict=result.verdict.value)

        too_many_failures = len(flow.failures) > self.config.voice.max_failed_answers
        if too_many_failures:
            ids = ", ".join(r.question_id for r in flow.failures)
            self._reject(s, f"Voice answers did not match the ID ({ids})")
        elif flow.finished:
            if flow.passed:
                s.stage = Stage.COMPLETED
                s.outcome = Outcome.MANUAL_REVIEW if (s.needs_review or s.duplicate_id_seen) else Outcome.APPROVED
                s.log(self._clock(), "completed", outcome=s.outcome.value)
                self._release_runtime(s)
                self._persist(s)
            else:
                self._reject(s, "Consent not given")
        else:
            self._persist(s)
        return AnswerOutcome(result, False, flow.current if s.stage is Stage.VOICE else None, s)


__all__ = ["AnswerOutcome", "KycConfig", "KycEngine", "Verdict"]

"""Session state and the storable record.

A session has two halves:

* **runtime** (in memory only, never persisted): the extracted ID fields, the ID photo,
  the voice-flow state. These contain PII and images.
* **record** (``to_record()``): what gets stored - stage, verdicts, scores, a *masked*
  ID number plus keyed hash, and an audit trail. No name, DOB, address, transcripts or images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Stage(str, Enum):
    CREATED = "created"
    DOCUMENT = "document"  # waiting for the ID upload
    LIVENESS = "liveness"  # ID accepted; waiting for the live blink check
    VOICE = "voice"  # face matched; voice questions in progress
    COMPLETED = "completed"
    REJECTED = "rejected"


class Outcome(str, Enum):
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"


@dataclass
class AuditEvent:
    ts: datetime
    stage: str
    event: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class KycSession:
    id: str
    created_at: datetime
    stage: Stage = Stage.DOCUMENT
    outcome: Outcome = Outcome.IN_PROGRESS
    rejection_reason: str | None = None
    needs_review: bool = False
    video_attempts: int = 0

    # persisted summary (no raw PII)
    doc_type: str | None = None
    doc_confidence: float | None = None
    ocr_confidence: float | None = None
    id_masked: str | None = None
    id_hash: str | None = None
    duplicate_id_seen: bool = False
    doc_warnings: list[str] = field(default_factory=list)  # e.g. 'Could not read name' (no PII)
    blinks: int | None = None
    face: dict[str, Any] | None = None
    voice: list[dict[str, Any]] = field(default_factory=list)
    events: list[AuditEvent] = field(default_factory=list)

    # runtime-only (never persisted)
    fields: Any = None  # document.extract.IdFields
    doc_photo: Any = None  # face.pipeline.Face
    voice_flow: Any = None  # voice.flow.VoiceFlow

    @property
    def terminal(self) -> bool:
        return self.stage in (Stage.COMPLETED, Stage.REJECTED)

    def log(self, ts: datetime, event: str, **detail: Any) -> None:
        self.events.append(AuditEvent(ts, self.stage.value, event, detail))

    def to_record(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "created_at": self.created_at,
            "stage": self.stage.value,
            "outcome": self.outcome.value,
            "rejection_reason": self.rejection_reason,
            "needs_review": self.needs_review,
            "doc_type": self.doc_type,
            "doc_confidence": self.doc_confidence,
            "ocr_confidence": self.ocr_confidence,
            "id_masked": self.id_masked,
            "id_hash": self.id_hash,
            "duplicate_id_seen": self.duplicate_id_seen,
            "doc_warnings": self.doc_warnings,
            "blinks": self.blinks,
            "face": self.face,
            "voice": self.voice,
            "events": [
                {"ts": e.ts, "stage": e.stage, "event": e.event, "detail": e.detail} for e in self.events
            ],
        }

    def public_view(self) -> dict[str, Any]:
        """JSON-safe summary for API responses / UI (same PII rules as the stored record)."""
        rec = self.to_record()
        rec["created_at"] = self.created_at.isoformat()
        rec["events"] = [{**e, "ts": e["ts"].isoformat()} for e in rec["events"]]
        return rec

"""REST API for the full Video-KYC flow.  Run with:  uvicorn videokyc.api:app --port 8000

    POST /v1/sessions                         start a session
    POST /v1/sessions/{id}/document           upload ID (PDF/JPG/PNG) [+ password]  -> OCR, classify, photo
    POST /v1/sessions/{id}/video              upload a short webcam clip           -> blink liveness + face match
    GET  /v1/sessions/{id}/question           current voice question
    POST /v1/sessions/{id}/answer             {"transcript": "..."}
    POST /v1/sessions/{id}/answer/audio       audio file, transcribed offline with Whisper
    GET  /v1/sessions/{id}                    PII-free session summary + audit trail
    POST /v1/face/verify                      stateless ID-photo vs selfie comparison

Uploads are processed in memory; only a masked, PII-free record is persisted.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__
from .document import intake
from .document.intake import MAX_BYTES
from .errors import (
    DocumentError,
    KycError,
    LivenessFailedError,
    LowQualityError,
    NoFaceError,
    OcrUnavailableError,
    PasswordRequiredError,
    SessionNotFoundError,
    SessionStateError,
    SpoofDetectedError,
    UnsupportedDocumentError,
    WrongPasswordError,
)
from .factory import build_engine
from .session.engine import KycEngine

app = FastAPI(
    title="Video KYC API",
    version=__version__,
    description="ID OCR + classification, live blink liveness, face verification and voice KYC. "
    "Files are processed in memory; only masked outcomes are stored.",
)

# (exception, HTTP status, machine-readable code); subclasses must precede their parents.
_ERRORS: list[tuple[type[KycError], int, str]] = [
    (PasswordRequiredError, 401, "password_required"),
    (WrongPasswordError, 401, "wrong_password"),
    (DocumentError, 400, "invalid_document"),
    (UnsupportedDocumentError, 422, "unsupported_document"),
    (NoFaceError, 422, "no_face_detected"),
    (LowQualityError, 422, "low_quality_image"),
    (LivenessFailedError, 422, "liveness_failed"),
    (SpoofDetectedError, 403, "spoof_detected"),
    (SessionNotFoundError, 404, "session_not_found"),
    (SessionStateError, 409, "invalid_session_state"),
    (OcrUnavailableError, 503, "ocr_unavailable"),
]


@lru_cache(maxsize=1)
def get_engine() -> KycEngine:
    return build_engine()


Engine = Annotated[KycEngine, Depends(get_engine)]


@app.exception_handler(KycError)
async def _handle_kyc_error(_, exc: KycError) -> JSONResponse:
    for cls, status, code in _ERRORS:
        if isinstance(exc, cls):
            body: dict = {"error": code, "detail": str(exc)}
            if isinstance(exc, LowQualityError):
                body["issues"] = exc.issues
            if isinstance(exc, LivenessFailedError):
                body["attempts_left"] = exc.attempts_left
            return JSONResponse(body, status_code=status)
    return JSONResponse({"error": "kyc_error", "detail": str(exc)}, status_code=400)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/v1/sessions", status_code=201)
def create_session(engine: Engine) -> dict:
    s = engine.create_session()
    return {"session_id": s.id, "stage": s.stage.value}


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str, engine: Engine) -> dict:
    return engine.get(session_id).public_view()


@app.post("/v1/sessions/{session_id}/document")
async def upload_document(
    session_id: str,
    engine: Engine,
    file: Annotated[UploadFile, File(description="Government ID: PDF, JPEG or PNG")],
    password: Annotated[str | None, Form(description="PDF password, if the file is encrypted")] = None,
) -> dict:
    data = await file.read(MAX_BYTES + 1)
    s = await run_in_threadpool(engine.submit_document, session_id, data, password)
    return {
        "stage": s.stage.value, "doc_type": s.doc_type, "doc_confidence": s.doc_confidence,
        "ocr_confidence": s.ocr_confidence, "id_masked": s.id_masked,
        "duplicate_id_seen": s.duplicate_id_seen,
        "warnings": s.doc_warnings,
    }


@app.post("/v1/sessions/{session_id}/video")
async def upload_video(
    session_id: str, engine: Engine, video: Annotated[UploadFile, File(description="Short webcam clip (mp4/webm)")]
) -> dict:
    data = await video.read(MAX_BYTES * 4 + 1)
    s = await run_in_threadpool(engine.submit_video, session_id, data)
    return {"stage": s.stage.value, "blinks": s.blinks, "face": s.face, "needs_review": s.needs_review,
            "rejection_reason": s.rejection_reason}


def _question_payload(engine: KycEngine, session_id: str, q) -> dict | None:
    if q is None:
        return None
    flow = engine.get(session_id).voice_flow
    return {"id": q.id, "prompt": q.prompt, "kind": q.kind, "index": flow.index + 1 if flow else None,
            "total": len(flow.questions) if flow else None, "attempts_left": flow.attempts_left if flow else None}


@app.get("/v1/sessions/{session_id}/question")
def current_question(session_id: str, engine: Engine) -> dict:
    return {"question": _question_payload(engine, session_id, engine.current_question(session_id))}


class Answer(BaseModel):
    transcript: str


def _answer_response(engine: KycEngine, session_id: str, out) -> dict:
    return {
        "question_id": out.result.question_id, "verdict": out.result.verdict.value, "detail": out.result.detail,
        "retry": out.retry, "stage": out.session.stage.value, "outcome": out.session.outcome.value,
        "rejection_reason": out.session.rejection_reason,
        "next_question": _question_payload(engine, session_id, out.next_question)
        if out.session.voice_flow else None,
    }


@app.post("/v1/sessions/{session_id}/answer")
def answer(session_id: str, body: Answer, engine: Engine) -> dict:
    return _answer_response(engine, session_id, engine.submit_answer(session_id, transcript=body.transcript))


@app.post("/v1/sessions/{session_id}/answer/audio")
async def answer_audio(
    session_id: str, engine: Engine, audio: Annotated[UploadFile, File(description="WAV/MP3/WebM answer")]
) -> dict:
    data = await audio.read(MAX_BYTES + 1)
    out = await run_in_threadpool(engine.submit_answer, session_id, None, data)
    return _answer_response(engine, session_id, out)


@app.post("/v1/face/verify")
async def verify_faces(
    document: Annotated[UploadFile, File(description="ID image/PDF")],
    selfie: Annotated[UploadFile, File(description="Selfie image")],
    engine: Engine,
    password: Annotated[str | None, Form()] = None,
) -> dict:
    """Stateless ID-photo vs selfie comparison (no session, nothing stored)."""
    doc_img = intake.load_pages(await document.read(MAX_BYTES + 1), password, engine.face.config.pdf_dpi)[0]
    selfie_img = intake.load_image(await selfie.read(MAX_BYTES + 1))
    result = await run_in_threadpool(engine.face.verify_images, doc_img, selfie_img)
    return result.to_dict()

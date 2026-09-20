"""REST API.  Run with:  uvicorn facematch.api:app --port 8000"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import __version__
from .config import Config
from .errors import (
    DocumentError,
    FaceMatchError,
    LowQualityError,
    NoFaceError,
    PasswordRequiredError,
    SpoofDetectedError,
    WrongPasswordError,
)
from .ingest import MAX_BYTES
from .pipeline import FaceMatchPipeline

app = FastAPI(
    title="Aadhaar Face Match API",
    version=__version__,
    description="Verify a live selfie against the photo on an Aadhaar PDF/image. "
    "Files are processed in memory and never stored.",
)

# (status code, machine-readable code) per error type; order matters (subclasses first).
_ERRORS: list[tuple[type[FaceMatchError], int, str]] = [
    (PasswordRequiredError, 401, "password_required"),
    (WrongPasswordError, 401, "wrong_password"),
    (DocumentError, 400, "invalid_document"),
    (NoFaceError, 422, "no_face_detected"),
    (LowQualityError, 422, "low_quality_image"),
    (SpoofDetectedError, 403, "spoof_detected"),
]


@lru_cache(maxsize=1)
def get_pipeline() -> FaceMatchPipeline:
    return FaceMatchPipeline(Config.from_env())


@app.exception_handler(FaceMatchError)
async def _handle_facematch_error(_, exc: FaceMatchError) -> JSONResponse:
    for cls, status, code in _ERRORS:
        if isinstance(exc, cls):
            body = {"error": code, "detail": str(exc)}
            if isinstance(exc, LowQualityError):
                body["issues"] = exc.issues
            return JSONResponse(body, status_code=status)
    return JSONResponse({"error": "verification_failed", "detail": str(exc)}, status_code=400)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/verify")
async def verify(
    document: Annotated[UploadFile, File(description="Aadhaar PDF, JPEG or PNG")],
    selfie: Annotated[UploadFile, File(description="Live capture, JPEG or PNG")],
    pipeline: Annotated[FaceMatchPipeline, Depends(get_pipeline)],
    password: Annotated[
        str | None,
        Form(description="PDF password (e-Aadhaar: first 4 letters of name in CAPS + birth year)"),
    ] = None,
) -> dict:
    doc_bytes = await document.read(MAX_BYTES + 1)
    selfie_bytes = await selfie.read(MAX_BYTES + 1)
    result = await run_in_threadpool(pipeline.verify, doc_bytes, selfie_bytes, password)
    return result.to_dict()

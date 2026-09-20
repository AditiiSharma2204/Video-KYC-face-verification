"""Exception hierarchy. Every expected failure maps to one of these so callers
(API, UI) can turn them into precise user-facing messages."""

from __future__ import annotations


class KycError(Exception):
    """Base class for all expected pipeline failures."""


# Kept as an alias so face-level code/tests read naturally.
FaceMatchError = KycError


# -- document ---------------------------------------------------------------


class DocumentError(KycError):
    """The uploaded document could not be read (corrupt, unsupported, empty)."""


class PasswordRequiredError(DocumentError):
    """The PDF is encrypted and no password was supplied."""


class WrongPasswordError(DocumentError):
    """The PDF is encrypted and the supplied password is incorrect."""


class OcrUnavailableError(KycError):
    """No OCR engine could be started (e.g. the Tesseract binary is missing)."""


class UnsupportedDocumentError(KycError):
    """The document could not be recognised as one of the supported government IDs."""


# -- face -------------------------------------------------------------------


class NoFaceError(KycError):
    """No face was found in the given image."""

    def __init__(self, source: str):
        super().__init__(f"No face detected in {source}.")
        self.source = source


class LowQualityError(KycError):
    """The face image is too poor (blur, size, lighting) for a reliable match."""

    def __init__(self, source: str, issues: list[str]):
        super().__init__(f"{source} image quality too low: {', '.join(issues)}.")
        self.source = source
        self.issues = issues


class SpoofDetectedError(KycError):
    """Presentation-attack (photo/screen replay) detected on the live capture."""


# -- session ----------------------------------------------------------------


class SessionStateError(KycError):
    """An action was attempted out of order (e.g. voice KYC before face match)."""


class SessionNotFoundError(KycError):
    """No KYC session exists with the given id."""


class LivenessFailedError(KycError):
    """The live check did not complete (no blink detected, or no clear face frames)."""

    def __init__(self, message: str, attempts_left: int):
        super().__init__(message)
        self.attempts_left = attempts_left

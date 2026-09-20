"""Exception hierarchy. Every expected failure maps to one of these so callers
(API, UI) can turn them into precise user-facing messages."""

from __future__ import annotations


class FaceMatchError(Exception):
    """Base class for all expected pipeline failures."""


class DocumentError(FaceMatchError):
    """The uploaded document could not be read (corrupt, unsupported, empty)."""


class PasswordRequiredError(DocumentError):
    """The PDF is encrypted and no password was supplied."""


class WrongPasswordError(DocumentError):
    """The PDF is encrypted and the supplied password is incorrect."""


class NoFaceError(FaceMatchError):
    """No face was found in the given image."""

    def __init__(self, source: str):
        super().__init__(f"No face detected in {source}.")
        self.source = source


class LowQualityError(FaceMatchError):
    """The face image is too poor (blur, size, lighting) for a reliable match."""

    def __init__(self, source: str, issues: list[str]):
        super().__init__(f"{source} image quality too low: {', '.join(issues)}.")
        self.source = source
        self.issues = issues


class SpoofDetectedError(FaceMatchError):
    """Presentation-attack (photo/screen replay) detected on the live capture."""

"""Speech-to-text backends. Offline by default (faster-whisper); no audio leaves the machine."""

from __future__ import annotations

import io
from typing import Protocol


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes) -> str:
        """Transcribe a WAV/MP3/WebM audio clip to plain text."""
        ...


class FasterWhisperStt:
    """Whisper via CTranslate2 (CPU int8). ``base.en`` is ~140 MB and downloaded once on first use.

    The initial prompt biases decoding toward the kind of content KYC answers contain
    (digits, spelled-out IDs, dates, Indian names) which measurably helps on numbers.
    """

    PROMPT = "Aadhaar, PAN, date of birth, 1990, 12 August, one two three four, A B C D E, Kumar, Sharma, India."

    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes) -> str:
        segments, _ = self._model.transcribe(
            io.BytesIO(audio),
            language="en",
            beam_size=5,
            vad_filter=True,
            initial_prompt=self.PROMPT,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

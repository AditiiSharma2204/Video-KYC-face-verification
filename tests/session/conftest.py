from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tests.conftest import FakeDetector, png_bytes, textured
from videokyc.document import DocumentProcessor
from videokyc.document.ocr import OcrResult
from videokyc.face import FaceConfig, FaceMatchPipeline
from videokyc.session.engine import KycConfig, KycEngine
from videokyc.storage.memory import MemoryStore

OPEN, CLOSED = 0.30, 0.08

PAN_TEXT = """INCOME TAX DEPARTMENT
GOVT. OF INDIA
Permanent Account Number
ABCPE1234F
Name
RAHUL KUMAR SHARMA
Father's Name
SURESH KUMAR SHARMA
Date of Birth
15/08/1990
Signature
"""

PERFECT_ANSWERS = [
    "thirty six", "yes", "Rahul Kumar Sharma", "15th August 1990", "A B C P E 1 2 3 4 F", "male",
    "no I have not", "yes", "I consent",
]  # no address on a PAN, so that question is 'recorded' with any 3+ word answer
ADDRESS_ANSWER = "flat twelve MG Road Bengaluru"


class FakeOcr:
    def __init__(self, text: str = PAN_TEXT):
        self.text = text

    def read(self, img):
        return OcrResult(self.text, 88.0)


class SeqEmbedder:
    """First embed() call is the ID photo, later calls are live frames."""

    model_name = "fake"

    def __init__(self, doc_vec, live_vec):
        self.doc, self.live, self.calls = np.asarray(doc_vec, float), np.asarray(live_vec, float), 0

    def embed(self, face):
        self.calls += 1
        v = self.doc if self.calls == 1 else self.live
        return v / np.linalg.norm(v)


class ScriptedEyes:
    def __init__(self, ears):
        self.ears = list(ears)

    def ear(self, frame):
        return self.ears.pop(0) if self.ears else OPEN


def blink_script(n=2, warm=15):
    ears = [OPEN] * warm
    for _ in range(n):
        ears += [CLOSED] * 4 + [OPEN] * 15
    return ears


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class FakeStt:
    def __init__(self, text="yes"):
        self.text, self.seen = text, []

    def transcribe(self, audio):
        self.seen.append(audio)
        return self.text


class Harness:
    """Wires a KycEngine from fakes and offers helpers to drive each stage."""

    def __init__(self, *, embedder=None, ocr_text=PAN_TEXT, config=None, stt=None, liveness=None):
        self.clock = Clock()
        self.store = MemoryStore()
        self.ocr = FakeOcr(ocr_text)
        self.eye_scripts = [blink_script(2)]
        self.face = FaceMatchPipeline(
            FaceConfig(model_name="ArcFace"), FakeDetector(), embedder or SeqEmbedder([1, 0], [1, 0.05]), liveness
        )
        self.engine = KycEngine(
            DocumentProcessor(self.ocr, self.face), self.face, self._eyes, self.store, stt,
            config or KycConfig(), self.clock,
        )

    def _eyes(self):
        return ScriptedEyes(self.eye_scripts.pop(0) if self.eye_scripts else blink_script(2))

    # helpers -------------------------------------------------------------------------
    def doc(self, sid, password=None, data=None):
        return self.engine.submit_document(sid, data or png_bytes(textured(1)), password)

    def live(self, sid, n_frames=None):
        cap = self.engine.new_capture(sid)
        n = n_frames or len(cap.eyes.ears)
        for i in range(n):
            cap.process(textured(100 + i), i / 30)
        return self.engine.complete_liveness(sid, cap)

    def to_voice(self):
        s = self.engine.create_session()
        self.doc(s.id)
        self.live(s.id)
        return s

    def answer_all(self, sid, answers=None):
        out = None
        for a in answers or PERFECT_ANSWERS_WITH_ADDRESS:
            out = self.engine.submit_answer(sid, a)
        return out


PERFECT_ANSWERS_WITH_ADDRESS = PERFECT_ANSWERS[:6] + [ADDRESS_ANSWER] + PERFECT_ANSWERS[6:]


@pytest.fixture
def harness():
    return Harness()

"""Live capture stage: face detection + blink liveness + quality gate, frame by frame.

Feed it frames (from a WebRTC stream or a decoded video); when the blink challenge
is satisfied, ``best_frames()`` returns the sharpest, eyes-open, quality-passing
faces to hand to the face matcher.
"""

from __future__ import annotations

import heapq
import itertools
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ..errors import DocumentError
from ..face import detection, quality
from ..face.models import FaceBox
from ..face.pipeline import Face, FaceMatchPipeline
from .blink import BlinkConfig, BlinkDetector
from .landmarks import EyeTracker

OPEN_EYE_FRACTION = 0.9  # only keep candidate frames whose EAR is near the open-eye baseline


@dataclass
class FrameStatus:
    face_found: bool = False
    ear: float | None = None
    blinked: bool = False
    blinks: int = 0
    quality_ok: bool = False
    issues: list[str] = field(default_factory=list)
    box: FaceBox | None = None


@dataclass
class Candidate:
    face: Face
    frame: np.ndarray
    score: float


class LiveCapture:
    def __init__(
        self,
        face: FaceMatchPipeline,
        eyes: EyeTracker,
        blink: BlinkConfig | None = None,
        keep: int = 5,
    ) -> None:
        self.face = face
        self.eyes = eyes
        self.blink = BlinkDetector(blink or BlinkConfig())
        self.keep = keep
        self._heap: list[tuple[float, int, Candidate]] = []  # min-heap on score
        self._tick = itertools.count()
        self.frames_seen = 0

    @property
    def blink_passed(self) -> bool:
        return self.blink.passed

    @property
    def timed_out(self) -> bool:
        return self.blink.timed_out

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameStatus:
        self.frames_seen += 1
        ear = self.eyes.ear(frame_bgr)
        blinked = self.blink.update(ear, t)
        status = FrameStatus(ear=ear, blinked=blinked, blinks=self.blink.blinks)
        if ear is None:
            return status

        box = detection.best_face(self.face.detector.detect(frame_bgr))
        if box is None:
            return status
        status.face_found, status.box = True, box

        report = quality.assess(frame_bgr, box, self.face.config)
        status.quality_ok, status.issues = report.passed, report.issues

        base = self.blink.baseline
        eyes_open = base is not None and ear >= base * OPEN_EYE_FRACTION
        if report.passed and eyes_open:
            crop = detection.align_face(frame_bgr, box, self.face.config)
            self._offer(Candidate(Face(crop, box, report), frame_bgr.copy(), report.sharpness))
        return status

    def _offer(self, cand: Candidate) -> None:
        item = (cand.score, next(self._tick), cand)
        if len(self._heap) < self.keep:
            heapq.heappush(self._heap, item)
        elif cand.score > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def best_frames(self) -> list[Candidate]:
        return [c for _, _, c in sorted(self._heap, reverse=True)]

    def reset(self) -> None:
        self.blink.reset()
        self._heap.clear()
        self.frames_seen = 0


def iter_video_frames(
    data: bytes, max_fps: float = 10.0, max_seconds: float = 20.0
) -> Iterator[tuple[np.ndarray, float]]:
    """Decode an uploaded video (mp4/webm/...) to (BGR frame, timestamp-seconds), sub-sampled to ``max_fps``.

    OpenCV can only open files, so the bytes are spooled to a temp file that is
    deleted as soon as decoding finishes.
    """
    import cv2

    fd, path = tempfile.mkstemp(suffix=".video")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise DocumentError("Could not decode the video; expected mp4 or webm.")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, round(fps / max_fps))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = idx / fps
                if t > max_seconds:
                    break
                if idx % step == 0:
                    yield frame, t
                idx += 1
        finally:
            cap.release()
    finally:
        os.unlink(path)


def run_video(capture: LiveCapture, data: bytes, **kw) -> LiveCapture:
    """Run a whole video through ``capture`` and return it (stops early once both goals are met)."""
    for frame, t in iter_video_frames(data, **kw):
        capture.process(frame, t)
        if capture.blink_passed and len(capture.best_frames()) >= capture.keep:
            break
    return capture

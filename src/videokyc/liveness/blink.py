"""Blink-based liveness.

The eye aspect ratio (EAR, Soukupova & Cech 2016) collapses toward 0 when an eyelid
closes:

        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

A blink is a short dip: EAR falls below a *per-person* threshold, stays there for a
plausible duration, then recovers. A still photograph never dips; a person staring
at the camera for ten seconds does not either.

Design choices that matter in practice:
* the threshold adapts to the user's own open-eye baseline (glasses, squinting and
  eye shape shift raw EAR a lot), instead of a fixed 0.2;
* timing uses timestamps, not frame counts, so 10 fps and 30 fps webcams behave alike;
* an eyes-shut period longer than ``max_closed_s`` is not counted as a blink.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# MediaPipe FaceMesh landmark indices, ordered p1..p6 as in the EAR formula.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR for six (x, y) points ordered p1..p6."""
    eye = np.asarray(eye, dtype=float)
    if eye.shape != (6, 2):
        raise ValueError(f"expected 6 (x, y) points, got shape {eye.shape}")
    horizontal = np.linalg.norm(eye[0] - eye[3])
    if horizontal == 0:
        return 0.0
    return float((np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (2 * horizontal))


def mean_ear(landmarks: np.ndarray) -> float:
    """Average EAR of both eyes from an (N, 2) landmark array in FaceMesh ordering."""
    return (eye_aspect_ratio(landmarks[list(RIGHT_EYE)]) + eye_aspect_ratio(landmarks[list(LEFT_EYE)])) / 2


@dataclass
class BlinkConfig:
    required_blinks: int = 2
    close_ratio: float = 0.75  # "closed" when EAR < baseline * close_ratio
    min_closed_s: float = 0.04  # shorter dips are landmark jitter
    max_closed_s: float = 0.8  # longer is eyes shut, not a blink
    baseline_window: int = 90  # recent EAR samples used for the open-eye baseline
    warmup_samples: int = 8  # need this many samples before judging
    min_baseline: float = 0.15  # below this the eyes look permanently closed: don't trust it
    timeout_s: float = 12.0


@dataclass
class BlinkDetector:
    config: BlinkConfig = field(default_factory=BlinkConfig)
    blinks: int = 0
    _ears: deque = field(default_factory=deque)
    _closed_since: float | None = None
    _started: float | None = None
    _last_t: float | None = None

    @property
    def baseline(self) -> float | None:
        if len(self._ears) < self.config.warmup_samples:
            return None
        return float(np.percentile(self._ears, 80))  # open-eye level, robust to blink dips

    @property
    def passed(self) -> bool:
        return self.blinks >= self.config.required_blinks

    @property
    def timed_out(self) -> bool:
        return (
            not self.passed
            and self._started is not None
            and self._last_t is not None
            and self._last_t - self._started > self.config.timeout_s
        )

    def update(self, ear: float | None, t: float) -> bool:
        """Feed one frame (``ear`` is None when no face was found). Returns True on a new blink."""
        c = self.config
        if self._started is None:
            self._started = t
        self._last_t = t

        if ear is None:  # face lost: abandon any half-finished blink
            self._closed_since = None
            return False

        base = self.baseline
        closed = base is not None and base >= c.min_baseline and ear < base * c.close_ratio

        if closed:
            if self._closed_since is None:
                self._closed_since = t
            return False  # closed-eye frames stay out of the baseline window

        # Eyes open (or still warming up).
        blinked = False
        if self._closed_since is not None:
            duration = t - self._closed_since
            if c.min_closed_s <= duration <= c.max_closed_s:
                self.blinks += 1
                blinked = True
            self._closed_since = None

        self._ears.append(ear)
        while len(self._ears) > c.baseline_window:
            self._ears.popleft()
        return blinked

    def reset(self) -> None:
        self.blinks = 0
        self._ears.clear()
        self._closed_since = self._started = self._last_t = None

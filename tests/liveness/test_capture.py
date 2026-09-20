import cv2
import numpy as np
import pytest

from tests.conftest import FakeDetector, FakeEmbedder, textured
from videokyc.errors import DocumentError
from videokyc.face import FaceConfig, FaceMatchPipeline
from videokyc.face.models import FaceBox
from videokyc.liveness.blink import BlinkConfig
from videokyc.liveness.capture import LiveCapture, iter_video_frames, run_video

OPEN, CLOSED = 0.30, 0.08


class ScriptedEyes:
    """Returns a scripted EAR per frame (None = no face)."""

    def __init__(self, ears):
        self.ears = list(ears)

    def ear(self, frame):
        return self.ears.pop(0)


def make_capture(ears, keep=3, detector=None, **blink):
    face = FaceMatchPipeline(FaceConfig(), detector or FakeDetector(), FakeEmbedder([1, 0]))
    return LiveCapture(face, ScriptedEyes(ears), BlinkConfig(**blink), keep=keep)


def drive(cap, n, fps=30.0, frame_fn=lambda i: textured(i)):
    statuses = []
    for i in range(n):
        statuses.append(cap.process(frame_fn(i), i / fps))
    return statuses


def blink_script(n_blinks=2):
    ears = [OPEN] * 15
    for _ in range(n_blinks):
        ears += [CLOSED] * 4 + [OPEN] * 15
    return ears


def test_blink_flow_reports_progress_and_passes():
    ears = blink_script(2)
    cap = make_capture(ears)
    statuses = drive(cap, len(ears))
    assert cap.blink_passed
    assert [s.blinks for s in statuses][-1] == 2
    assert sum(s.blinked for s in statuses) == 2
    assert all(s.face_found for s in statuses)


def test_keeps_only_the_sharpest_open_eye_frames():
    ears = blink_script(1)
    cap = make_capture(ears, keep=3)
    drive(cap, len(ears))
    best = cap.best_frames()
    assert len(best) == 3
    scores = [c.score for c in best]
    assert scores == sorted(scores, reverse=True)
    # frames captured while the eyes were closed must never be candidates
    assert all(c.frame is not None for c in best)


def test_closed_eye_frames_are_not_candidates():
    ears = [OPEN] * 12 + [CLOSED] * 6
    cap = make_capture(ears, keep=50)
    drive(cap, len(ears))
    assert 0 < len(cap.best_frames()) <= 12  # at most the 12 open frames, none of the closed ones


def test_no_face_frames_yield_nothing():
    cap = make_capture([None] * 10)
    statuses = drive(cap, 10)
    assert not any(s.face_found for s in statuses) and cap.best_frames() == []


def test_blurry_frames_fail_quality_and_are_not_kept():
    ears = [OPEN] * 20
    cap = make_capture(ears)
    drive(cap, 20, frame_fn=lambda i: np.full((240, 240, 3), 128, np.uint8))  # featureless => "blurry"
    assert cap.best_frames() == []


def test_tiny_face_reports_quality_issue():
    cap = make_capture([OPEN] * 5, detector=FakeDetector([FaceBox(10, 10, 20, 20, 0.9)]))
    statuses = drive(cap, 5)
    assert not statuses[-1].quality_ok and any("too small" in i for i in statuses[-1].issues)


def test_reset_clears_state():
    ears = blink_script(1) * 2
    cap = make_capture(ears)
    drive(cap, len(blink_script(1)))
    cap.reset()
    assert cap.blink.blinks == 0 and cap.best_frames() == [] and cap.frames_seen == 0


# -- video decoding ---------------------------------------------------------------------


def write_avi(path, n_frames=45, fps=30, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        w.write(textured(i, size=size[0])[: size[1]])  # sharp noise so frames pass the quality gate
    w.release()
    return path.read_bytes()


def test_iter_video_frames_subsamples_and_timestamps(tmp_path):
    data = write_avi(tmp_path / "v.avi", n_frames=60, fps=30)
    frames = list(iter_video_frames(data, max_fps=10))
    assert 18 <= len(frames) <= 22  # 60 frames @30fps => ~20 @10fps
    ts = [t for _, t in frames]
    assert ts == sorted(ts) and ts[0] == 0 and ts[1] == pytest.approx(0.1, abs=0.04)
    assert frames[0][0].shape == (240, 320, 3)


def test_iter_video_frames_respects_max_seconds(tmp_path):
    data = write_avi(tmp_path / "v.avi", n_frames=90, fps=30)
    assert max(t for _, t in iter_video_frames(data, max_fps=30, max_seconds=1.0)) <= 1.0


def test_undecodable_video_raises_document_error():
    with pytest.raises(DocumentError):
        list(iter_video_frames(b"not a video at all"))


def test_run_video_stops_once_goals_met(tmp_path):
    data = write_avi(tmp_path / "v.avi", n_frames=90, fps=30)
    ears = blink_script(2) + [OPEN] * 200
    cap = make_capture(ears, keep=2)
    run_video(cap, data, max_fps=30)
    assert cap.blink_passed
    assert cap.frames_seen < 90  # early exit

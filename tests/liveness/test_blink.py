import numpy as np
import pytest

from videokyc.liveness.blink import LEFT_EYE, RIGHT_EYE, BlinkConfig, BlinkDetector, eye_aspect_ratio, mean_ear

OPEN, CLOSED = 0.30, 0.08


def eye(width=30.0, height=9.0):
    """Six eye landmarks p1..p6 for an eye of given width/height."""
    h = height / 2
    return np.array(
        [[0, 0], [width * 0.33, -h], [width * 0.66, -h], [width, 0], [width * 0.66, h], [width * 0.33, h]], float
    )


def feed(det, ears, fps=30.0, t0=0.0):
    t = t0
    for e in ears:
        det.update(e, t)
        t += 1 / fps
    return t


def blink_pattern(n_blinks, open_frames=20, closed_frames=4, warm=15):
    out = [OPEN] * warm
    for _ in range(n_blinks):
        out += [CLOSED] * closed_frames + [OPEN] * open_frames
    return out


# -- EAR geometry ----------------------------------------------------------------


def test_ear_scales_with_eye_openness_and_is_size_invariant():
    assert eye_aspect_ratio(eye(height=9)) == pytest.approx(0.30)
    assert eye_aspect_ratio(eye(height=2)) == pytest.approx(0.0667, abs=1e-3)
    assert eye_aspect_ratio(eye(width=60, height=18)) == pytest.approx(eye_aspect_ratio(eye(height=9)))


def test_ear_validates_shape_and_degenerate_eye():
    with pytest.raises(ValueError):
        eye_aspect_ratio(np.zeros((5, 2)))
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def test_mean_ear_uses_both_eyes_from_facemesh_indices():
    pts = np.zeros((478, 2))
    pts[list(RIGHT_EYE)] = eye(height=9)
    pts[list(LEFT_EYE)] = eye(height=3)
    assert mean_ear(pts) == pytest.approx((0.30 + 0.10) / 2)


# -- blink detection -------------------------------------------------------------------


def test_counts_blinks_and_passes_at_required_number():
    det = BlinkDetector(BlinkConfig(required_blinks=2))
    feed(det, blink_pattern(1))
    assert det.blinks == 1 and not det.passed
    feed(det, blink_pattern(1, warm=0), t0=10)
    assert det.blinks == 2 and det.passed


def test_still_photo_never_blinks():
    det = BlinkDetector()
    feed(det, [OPEN + np.random.default_rng(0).normal(0, 0.005) for _ in range(300)])
    assert det.blinks == 0 and not det.passed


def test_eyes_held_shut_is_not_a_blink():
    det = BlinkDetector(BlinkConfig(max_closed_s=0.8))
    feed(det, [OPEN] * 15 + [CLOSED] * 60 + [OPEN] * 10, fps=30)  # 2 s shut
    assert det.blinks == 0


def test_single_frame_jitter_is_not_a_blink():
    det = BlinkDetector(BlinkConfig(min_closed_s=0.04))
    feed(det, [OPEN] * 15 + [CLOSED] + [OPEN] * 10, fps=10)  # one 100ms-apart sample: duration = 1 frame gap
    # at 10 fps a single closed sample lasts one frame interval (0.1s) - that IS >= min_closed, so a blink
    assert det.blinks == 1
    det2 = BlinkDetector(BlinkConfig(min_closed_s=0.04))
    feed(det2, [OPEN] * 15 + [CLOSED] + [OPEN] * 10, fps=60)  # 16 ms: below the minimum => jitter
    assert det2.blinks == 0


@pytest.mark.parametrize("fps", [10, 15, 30, 60])
def test_frame_rate_independent(fps):
    frames_closed = max(2, round(0.15 * fps))  # a 150 ms blink at every frame rate
    det = BlinkDetector()
    feed(det, blink_pattern(2, open_frames=int(fps * 0.7), closed_frames=frames_closed, warm=int(fps * 1.0)), fps=fps)
    assert det.blinks == 2


def test_adapts_to_low_ear_users_glasses_or_narrow_eyes():
    open_low, closed_low = 0.17, 0.05  # a fixed 0.2 threshold would call this permanently closed
    det = BlinkDetector()
    seq = [open_low] * 15 + [closed_low] * 4 + [open_low] * 15
    feed(det, seq)
    assert det.blinks == 1


def test_permanently_tiny_ear_is_not_trusted():
    det = BlinkDetector(BlinkConfig(min_baseline=0.15))
    feed(det, [0.05] * 20 + [0.01] * 4 + [0.05] * 20)
    assert det.blinks == 0


def test_face_lost_mid_blink_abandons_it():
    det = BlinkDetector()
    t = feed(det, [OPEN] * 15 + [CLOSED] * 2)
    t = feed(det, [None] * 5, t0=t)
    feed(det, [OPEN] * 10, t0=t)
    assert det.blinks == 0


def test_timeout_and_reset():
    det = BlinkDetector(BlinkConfig(timeout_s=5))
    feed(det, [OPEN] * 200, fps=30)  # ~6.7 s without a blink
    assert det.timed_out
    det.reset()
    assert det.blinks == 0 and not det.timed_out and det.baseline is None


def test_update_reports_the_frame_a_blink_completes():
    det = BlinkDetector()
    events = []
    t = 0.0
    for e in blink_pattern(1):
        events.append(det.update(e, t))
        t += 1 / 30
    assert sum(events) == 1

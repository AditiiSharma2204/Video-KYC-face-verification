import numpy as np
import pytest

from tests.conftest import FakeDetector, FakeLiveness, textured
from videokyc.errors import LowQualityError, NoFaceError, SpoofDetectedError
from videokyc.face.models import Decision

SAME = ([1, 0], [1, 0])
ORTHOGONAL = ([1, 0], [0, 1])
BORDERLINE = ([1, 0], [0.3, np.sqrt(1 - 0.09)])  # cosine distance 0.70 vs threshold 0.68


def run(pipe, doc=None, selfie=None):
    return pipe.verify_images(doc if doc is not None else textured(1),
                              selfie if selfie is not None else textured(2))


def test_match(make_pipeline):
    r = run(make_pipeline(SAME))
    assert r.decision is Decision.MATCH and r.verified
    assert r.distance == pytest.approx(0.0, abs=1e-9)
    assert r.threshold == 0.68 and r.model == "ArcFace" and r.detector == "fake"
    assert {"detect", "embed_and_match"} <= r.timings_ms.keys()


def test_no_match(make_pipeline):
    r = run(make_pipeline(ORTHOGONAL))
    assert r.decision is Decision.NO_MATCH and not r.verified


def test_borderline_goes_to_review(make_pipeline):
    r = run(make_pipeline(BORDERLINE))
    assert r.decision is Decision.REVIEW and not r.verified


def test_custom_threshold_changes_decision(make_pipeline):
    assert run(make_pipeline(BORDERLINE, threshold=1.0, review_margin=0)).decision is Decision.MATCH


def test_no_face_in_document(make_pipeline):
    with pytest.raises(NoFaceError, match="Aadhaar"):
        run(make_pipeline(SAME, detector=FakeDetector([])))


def test_blurry_selfie_rejected_but_blank_document_only_warns(make_pipeline):
    blank = np.full((240, 240, 3), 128, np.uint8)
    with pytest.raises(LowQualityError) as exc:
        run(make_pipeline(SAME), selfie=blank)
    assert "image is blurry" in exc.value.issues

    r = run(make_pipeline(SAME), doc=blank)  # document photo: warn, don't reject
    assert "image is blurry" in r.document_quality.issues


def test_quality_gate_can_be_disabled(make_pipeline):
    blank = np.full((240, 240, 3), 128, np.uint8)
    assert run(make_pipeline(SAME, enforce_quality=False), selfie=blank).verified


def test_liveness_pass_and_fail(make_pipeline):
    ok = run(make_pipeline(SAME, liveness=FakeLiveness(True)))
    assert ok.liveness_score == 0.9 and "liveness" in ok.timings_ms
    with pytest.raises(SpoofDetectedError):
        run(make_pipeline(SAME, liveness=FakeLiveness(False)))


def test_result_serialises_to_json_types(make_pipeline):
    import json

    d = run(make_pipeline(SAME)).to_dict()
    json.dumps(d)  # must not raise (no numpy scalars, enums)
    assert d["decision"] == "match" and d["verified"] is True

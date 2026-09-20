import json

import pytest

from tests.conftest import FakeLiveness
from tests.document.test_intake import make_pdf
from tests.session.conftest import PERFECT_ANSWERS_WITH_ADDRESS, FakeStt, Harness, SeqEmbedder, blink_script
from videokyc.errors import (
    DocumentError,
    LivenessFailedError,
    PasswordRequiredError,
    SessionNotFoundError,
    SessionStateError,
    SpoofDetectedError,
    UnsupportedDocumentError,
    WrongPasswordError,
)
from videokyc.session.engine import KycConfig
from videokyc.session.models import Outcome, Stage
from videokyc.voice.questions import Verdict, VoiceConfig

PII = ["Rahul", "Sharma", "Suresh", "ABCPE1234F", "15/08/1990", "1990-08-15"]


def test_full_happy_path_to_approval(harness):
    e = harness.engine
    s = e.create_session()
    assert s.stage is Stage.DOCUMENT

    s = harness.doc(s.id)
    assert s.stage is Stage.LIVENESS and s.doc_type == "pan" and s.id_masked == "XXXXXX234F"

    s = harness.live(s.id)
    assert s.stage is Stage.VOICE and s.blinks == 2 and s.face["decision"] == "match"

    out = harness.answer_all(s.id)
    assert out.session.stage is Stage.COMPLETED and out.session.outcome is Outcome.APPROVED
    assert [v["question"] for v in out.session.voice] == [
        "age", "citizenship", "name", "dob", "doc_number", "gender", "address",
        "previous_updates", "bank_linking", "consent",
    ]
    # PAN has no address, so that answer is recorded rather than verified
    assert next(v for v in out.session.voice if v["question"] == "address")["verdict"] == "recorded"


def test_persisted_record_contains_no_pii_and_runtime_is_released(harness):
    s = harness.to_voice()
    harness.answer_all(s.id)
    record = harness.store.get(s.id)
    blob = json.dumps(record, default=str)
    for secret in PII:
        assert secret not in blob, f"PII leaked into stored record: {secret}"
    assert record["id_masked"] == "XXXXXX234F" and len(record["id_hash"]) == 64
    assert record["outcome"] == "approved" and record["face"]["decision"] == "match"
    s = harness.engine.get(s.id)
    assert s.fields is None and s.doc_photo is None and s.voice_flow is None  # PII dropped from memory


def test_audit_trail_records_every_stage_in_order(harness):
    s = harness.to_voice()
    harness.answer_all(s.id)
    events = [ev["event"] for ev in harness.store.get(s.id)["events"]]
    for expected in ["session_created", "document_accepted", "liveness_passed", "face_matched", "completed"]:
        assert expected in events
    assert events.index("document_accepted") < events.index("liveness_passed") < events.index("face_matched")


# -- document stage -----------------------------------------------------------------------------


def test_password_protected_pdf_flow(harness):
    s = harness.engine.create_session()
    pdf = make_pdf(password="RAHU1990")
    with pytest.raises(PasswordRequiredError):
        harness.doc(s.id, data=pdf)
    with pytest.raises(WrongPasswordError):
        harness.doc(s.id, password="nope", data=pdf)
    assert harness.doc(s.id, password="RAHU1990", data=pdf).stage is Stage.LIVENESS


def test_unrecognised_document_is_recoverable_then_rejected_after_max_attempts(harness):
    harness.ocr.text = "grocery list: milk, eggs, bread"
    s = harness.engine.create_session()
    for _ in range(2):
        with pytest.raises(UnsupportedDocumentError):
            harness.doc(s.id)
        assert harness.engine.get(s.id).stage is Stage.DOCUMENT
    with pytest.raises(UnsupportedDocumentError):
        harness.doc(s.id)
    s = harness.engine.get(s.id)
    assert s.stage is Stage.REJECTED and "invalid document" in s.rejection_reason


def test_corrupt_upload_counts_as_attempt(harness):
    s = harness.engine.create_session()
    with pytest.raises(DocumentError):
        harness.doc(s.id, data=b"garbage")
    assert harness.engine.get(s.id).stage is Stage.DOCUMENT


def test_doc_type_allowlist(harness):
    h = Harness(config=KycConfig(allowed_doc_types=frozenset()))
    s = h.engine.create_session()
    with pytest.raises(UnsupportedDocumentError):
        h.doc(s.id)


def test_duplicate_id_is_flagged_and_forces_manual_review(harness):
    first = harness.to_voice()
    harness.answer_all(first.id)
    assert harness.engine.get(first.id).outcome is Outcome.APPROVED

    harness.eye_scripts = [[0.30] * 15 + [0.08] * 4 + [0.30] * 15 + [0.08] * 4 + [0.30] * 15]
    harness.face.embedder = SeqEmbedder([1, 0], [1, 0.05])
    second = harness.to_voice()
    assert harness.engine.get(second.id).duplicate_id_seen
    harness.answer_all(second.id)
    assert harness.engine.get(second.id).outcome is Outcome.MANUAL_REVIEW


# -- liveness / face stage ------------------------------------------------------------------------


def test_no_blink_is_recoverable_and_exhausts_attempts(harness):
    s = harness.engine.create_session()
    harness.doc(s.id)
    for left in (2, 1):
        harness.eye_scripts = [[0.30] * 60]  # staring, never blinking
        with pytest.raises(LivenessFailedError) as exc:
            harness.live(s.id)
        assert exc.value.attempts_left == left and "blink" in str(exc.value)
        assert harness.engine.get(s.id).stage is Stage.LIVENESS
    harness.eye_scripts = [[0.30] * 60]
    with pytest.raises(LivenessFailedError):
        harness.live(s.id)
    s = harness.engine.get(s.id)
    assert s.stage is Stage.REJECTED and "Liveness" in s.rejection_reason


def test_retry_after_failed_liveness_can_still_succeed(harness):
    s = harness.engine.create_session()
    harness.doc(s.id)
    harness.eye_scripts = [[0.30] * 60, ]
    with pytest.raises(LivenessFailedError):
        harness.live(s.id)
    harness.eye_scripts = [blink_script(2)]
    assert harness.live(s.id).stage is Stage.VOICE


def test_face_mismatch_rejects(harness):
    h = Harness(embedder=SeqEmbedder([1, 0], [0, 1]))  # orthogonal => different person
    s = h.engine.create_session()
    h.doc(s.id)
    s = h.live(s.id)
    assert s.stage is Stage.REJECTED and s.face["decision"] == "no_match" and "does not match" in s.rejection_reason


def test_borderline_face_continues_flagged_for_review(harness):
    import numpy as np

    h = Harness(embedder=SeqEmbedder([1, 0], [0.3, np.sqrt(1 - 0.09)]))  # distance 0.70 vs threshold 0.68
    s = h.to_voice()
    assert s.face["decision"] == "review" and s.needs_review
    out = h.answer_all(s.id)
    assert out.session.outcome is Outcome.MANUAL_REVIEW


def test_borderline_face_can_be_configured_to_reject():
    import numpy as np

    h = Harness(embedder=SeqEmbedder([1, 0], [0.3, np.sqrt(1 - 0.09)]), config=KycConfig(review_action="reject"))
    s = h.engine.create_session()
    h.doc(s.id)
    assert h.live(s.id).stage is Stage.REJECTED


def test_spoof_detected_on_live_frame_rejects():
    h = Harness(liveness=FakeLiveness(real=False))
    s = h.engine.create_session()
    h.doc(s.id)
    with pytest.raises(SpoofDetectedError):
        h.live(s.id)
    assert h.engine.get(s.id).stage is Stage.REJECTED


def test_video_upload_path(tmp_path):
    from tests.liveness.test_capture import write_avi

    h = Harness()
    s = h.engine.create_session()
    h.doc(s.id)
    h.eye_scripts = [[0.30] * 15 + ([0.08] * 4 + [0.30] * 15) * 2 + [0.30] * 100]
    s = h.engine.submit_video(s.id, write_avi(tmp_path / "v.avi", n_frames=120, fps=30))
    assert s.stage is Stage.VOICE and s.blinks == 2


# -- voice stage ------------------------------------------------------------------------------------


def test_wrong_answer_gets_retry_then_rejects_the_session_early(harness):
    s = harness.to_voice()
    out = harness.engine.submit_answer(s.id, "twenty")
    assert out.retry and out.next_question.id == "age" and out.session.stage is Stage.VOICE
    out = harness.engine.submit_answer(s.id, "twenty one")
    assert not out.retry and out.session.stage is Stage.REJECTED
    assert "age" in out.session.rejection_reason
    assert harness.store.get(s.id)["outcome"] == "rejected"


def test_unclear_then_correct_answer_continues(harness):
    s = harness.to_voice()
    assert harness.engine.submit_answer(s.id, "mumble").retry
    out = harness.engine.submit_answer(s.id, "thirty six")
    assert not out.retry and out.next_question.id == "citizenship"


def test_refusing_consent_rejects(harness):
    s = harness.to_voice()
    answers = PERFECT_ANSWERS_WITH_ADDRESS[:-1] + ["no", "no"]
    out = harness.answer_all(s.id, answers)
    assert out.session.stage is Stage.REJECTED


def test_failure_tolerance_config():
    h = Harness(config=KycConfig(voice=VoiceConfig(max_failed_answers=1)))
    s = h.to_voice()
    answers = list(PERFECT_ANSWERS_WITH_ADDRESS)
    answers[3:4] = ["16th August 1990", "17th August 1990"]  # date of birth wrong twice (a PAN can verify it)
    out = h.answer_all(s.id, answers)
    assert out.session.outcome is Outcome.APPROVED
    assert [v for v in out.session.voice if v["verdict"] == "fail"][0]["question"] == "dob"


def test_audio_answers_go_through_speech_to_text():
    stt = FakeStt("thirty six")
    h = Harness(stt=stt)
    s = h.to_voice()
    out = h.engine.submit_answer(s.id, audio=b"RIFF....")
    assert stt.seen == [b"RIFF...."] and out.result.verdict is Verdict.PASS


def test_audio_without_stt_engine_is_a_clear_error(harness):
    s = harness.to_voice()
    with pytest.raises(SessionStateError, match="speech-to-text"):
        harness.engine.submit_answer(s.id, audio=b"x")
    with pytest.raises(ValueError):
        harness.engine.submit_answer(s.id)


# -- state machine + lifecycle ----------------------------------------------------------------------


def test_actions_out_of_order_are_refused(harness):
    s = harness.engine.create_session()
    with pytest.raises(SessionStateError):
        harness.engine.new_capture(s.id)
    with pytest.raises(SessionStateError):
        harness.engine.submit_answer(s.id, "yes")
    harness.doc(s.id)
    with pytest.raises(SessionStateError):
        harness.doc(s.id)  # can't re-upload once accepted


def test_terminal_sessions_accept_nothing_more(harness):
    s = harness.to_voice()
    harness.engine.submit_answer(s.id, "twenty")
    harness.engine.submit_answer(s.id, "twenty one")  # rejected
    with pytest.raises(SessionStateError):
        harness.engine.submit_answer(s.id, "thirty six")


def test_unknown_session():
    with pytest.raises(SessionNotFoundError):
        Harness().engine.get("nope")


def test_sessions_expire(harness):
    s = harness.engine.create_session()
    harness.clock.advance(harness.engine.config.session_ttl_s + 1)
    with pytest.raises(SessionStateError):  # expiry rejects it, so the DOCUMENT stage is no longer valid
        harness.doc(s.id)
    assert harness.engine.get(s.id).rejection_reason == "Session expired"


def test_public_view_is_json_serialisable(harness):
    s = harness.to_voice()
    view = harness.engine.get(s.id).public_view()
    json.dumps(view)
    for secret in PII:
        assert secret not in json.dumps(view)

import pytest
from fastapi.testclient import TestClient

from tests.conftest import png_bytes, textured
from tests.document.test_intake import make_pdf
from tests.liveness.test_capture import write_avi
from tests.session.conftest import PERFECT_ANSWERS_WITH_ADDRESS, FakeStt, Harness, blink_script
from videokyc.api import app, get_engine


@pytest.fixture
def api():
    holder = {}

    def install(harness: Harness | None = None) -> tuple[TestClient, Harness]:
        h = harness or Harness(stt=FakeStt("thirty six"))
        holder["h"] = h
        app.dependency_overrides[get_engine] = lambda: h.engine
        return TestClient(app), h

    yield install
    app.dependency_overrides.clear()


def start(c):
    r = c.post("/v1/sessions")
    assert r.status_code == 201
    return r.json()["session_id"]


def upload_doc(c, sid, data=None, **form):
    return c.post(f"/v1/sessions/{sid}/document", files={"file": ("id.png", data or png_bytes(textured(1)))}, data=form)


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_full_flow_over_http(api, tmp_path):
    c, h = api()
    sid = start(c)

    r = upload_doc(c, sid)
    assert r.status_code == 200
    body = r.json()
    assert body["doc_type"] == "pan" and body["id_masked"] == "XXXXXX234F" and body["stage"] == "liveness"
    assert "ABCPE1234F" not in r.text and "Rahul" not in r.text

    h.eye_scripts = [[0.30] * 15 + ([0.08] * 4 + [0.30] * 15) * 2 + [0.30] * 100]
    r = c.post(f"/v1/sessions/{sid}/video", files={"video": ("v.avi", write_avi(tmp_path / "v.avi", 120))})
    assert r.status_code == 200 and r.json()["stage"] == "voice" and r.json()["face"]["decision"] == "match"

    q = c.get(f"/v1/sessions/{sid}/question").json()["question"]
    assert q["id"] == "age" and q["index"] == 1 and q["total"] == 10

    last = None
    for a in PERFECT_ANSWERS_WITH_ADDRESS:
        last = c.post(f"/v1/sessions/{sid}/answer", json={"transcript": a}).json()
    assert last["stage"] == "completed" and last["outcome"] == "approved" and last["next_question"] is None

    view = c.get(f"/v1/sessions/{sid}").json()
    assert view["outcome"] == "approved" and len(view["voice"]) == 10
    assert "Rahul" not in c.get(f"/v1/sessions/{sid}").text


def test_answer_retry_payload(api, tmp_path):
    c, h = api()
    sid = start(c)
    upload_doc(c, sid)
    h.eye_scripts = [blink_script(2) + [0.30] * 60]
    c.post(f"/v1/sessions/{sid}/video", files={"video": ("v.avi", write_avi(tmp_path / "v.avi", 120))})
    r = c.post(f"/v1/sessions/{sid}/answer", json={"transcript": "mumble"}).json()
    assert r["verdict"] == "unclear" and r["retry"] is True and r["next_question"]["attempts_left"] == 1


def test_audio_answer_uses_stt(api, tmp_path):
    c, h = api()
    sid = start(c)
    upload_doc(c, sid)
    h.eye_scripts = [blink_script(2) + [0.30] * 60]
    c.post(f"/v1/sessions/{sid}/video", files={"video": ("v.avi", write_avi(tmp_path / "v.avi", 120))})
    r = c.post(f"/v1/sessions/{sid}/answer/audio", files={"audio": ("a.wav", b"RIFFxxxx")})
    assert r.status_code == 200 and r.json()["verdict"] == "pass" and h.engine.stt.seen == [b"RIFFxxxx"]


def test_password_flow_status_codes(api):
    c, _ = api()
    sid = start(c)
    pdf = make_pdf(password="RAHU1990")
    r = upload_doc(c, sid, pdf)
    assert (r.status_code, r.json()["error"]) == (401, "password_required")
    r = upload_doc(c, sid, pdf, password="bad")
    assert (r.status_code, r.json()["error"]) == (401, "wrong_password")
    assert upload_doc(c, sid, pdf, password="RAHU1990").status_code == 200


def test_error_mapping(api):
    c, h = api()
    sid = start(c)
    assert c.get("/v1/sessions/nope").status_code == 404
    assert c.post("/v1/sessions/nope/answer", json={"transcript": "x"}).json()["error"] == "session_not_found"

    r = c.get(f"/v1/sessions/{sid}/question")  # wrong stage
    assert (r.status_code, r.json()["error"]) == (409, "invalid_session_state")

    assert upload_doc(c, sid, b"garbage").json()["error"] == "invalid_document"
    h.ocr.text = "grocery list"
    r = upload_doc(c, sid)
    assert (r.status_code, r.json()["error"]) == (422, "unsupported_document")


def test_liveness_failure_reports_attempts_left(api, tmp_path):
    c, h = api()
    sid = start(c)
    upload_doc(c, sid)
    h.eye_scripts = [[0.30] * 200]
    r = c.post(f"/v1/sessions/{sid}/video", files={"video": ("v.avi", write_avi(tmp_path / "v.avi", 120))})
    assert (r.status_code, r.json()["error"], r.json()["attempts_left"]) == (422, "liveness_failed", 2)


def test_validation_errors_are_422(api):
    c, _ = api()
    sid = start(c)
    assert c.post(f"/v1/sessions/{sid}/answer", json={}).status_code == 422
    assert c.post(f"/v1/sessions/{sid}/document").status_code == 422


def test_stateless_face_verify(api):
    c, _ = api()
    r = c.post("/v1/face/verify", files={"document": ("d.png", png_bytes(textured(1))),
                                         "selfie": ("s.png", png_bytes(textured(2)))})
    assert r.status_code == 200 and r.json()["decision"] == "match"

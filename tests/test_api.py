import pytest
from fastapi.testclient import TestClient

from facematch.api import app, get_pipeline

from .conftest import FakeDetector, png_bytes, textured
from .test_ingest import make_pdf


@pytest.fixture
def client(make_pipeline):
    def install(embeddings, **kw):
        app.dependency_overrides[get_pipeline] = lambda: make_pipeline(embeddings, **kw)
        return TestClient(app)

    yield install
    app.dependency_overrides.clear()


def post(c, doc, selfie=None, **data):
    return c.post(
        "/verify",
        files={"document": ("a.png", doc), "selfie": ("s.png", selfie or png_bytes(textured(2)))},
        data=data,
    )


def test_health():
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_verify_match(client):
    r = post(client([[1, 0], [1, 0]]), png_bytes(textured(1)))
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "match" and body["verified"] is True


def test_verify_encrypted_pdf_flow(client):
    pdf = make_pdf(password="RAHU1990")
    assert post(client([[1, 0], [1, 0]]), pdf).json()["error"] == "password_required"
    assert post(client([[1, 0], [1, 0]]), pdf, password="bad").json()["error"] == "wrong_password"
    r = post(client([[1, 0], [1, 0]]), pdf, password="RAHU1990")
    assert r.status_code == 200 and r.json()["verified"] is True


def test_error_status_codes(client):
    assert post(client([[1, 0]]), b"garbage").status_code == 400
    r = post(client([[1, 0]], detector=FakeDetector([])), png_bytes(textured(1)))
    assert r.status_code == 422 and r.json()["error"] == "no_face_detected"


def test_missing_selfie_is_422(client):
    r = client([[1, 0]]).post("/verify", files={"document": ("a.png", png_bytes(textured()))})
    assert r.status_code == 422

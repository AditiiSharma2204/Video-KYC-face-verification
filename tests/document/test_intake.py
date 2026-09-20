import pymupdf
import pytest

from tests.conftest import png_bytes, textured
from videokyc.document import intake as ingest
from videokyc.errors import DocumentError, PasswordRequiredError, WrongPasswordError


def make_pdf(password: str | None = None) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(50, 50, 250, 250), stream=png_bytes(textured()))
    if password:
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw="owner-secret")
    return doc.tobytes()


def test_is_pdf():
    assert ingest.is_pdf(make_pdf())
    assert not ingest.is_pdf(png_bytes(textured()))


def test_render_plain_pdf():
    img = ingest.load_document(make_pdf(), dpi=72)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 700  # A4-ish at 72 dpi


def test_encrypted_pdf_requires_and_accepts_password():
    data = make_pdf(password="RAHU1990")
    with pytest.raises(PasswordRequiredError):
        ingest.load_document(data)
    with pytest.raises(WrongPasswordError):
        ingest.load_document(data, password="nope")
    assert ingest.load_document(data, password="RAHU1990", dpi=72).shape[2] == 3


def test_image_document_loads():
    assert ingest.load_document(png_bytes(textured())).shape == (240, 240, 3)


@pytest.mark.parametrize("bad", [b"", b"not an image", b"%PDF-1.7 garbage"])
def test_bad_input_raises_document_error(bad):
    with pytest.raises(DocumentError):
        ingest.load_document(bad)


def test_size_limit(monkeypatch):
    monkeypatch.setattr(ingest, "MAX_BYTES", 10)
    with pytest.raises(DocumentError, match="limit"):
        ingest.load_image(b"x" * 11)

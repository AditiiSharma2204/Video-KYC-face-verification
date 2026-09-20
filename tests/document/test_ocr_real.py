"""Real Tesseract, on synthetic cards. Skipped automatically where Tesseract isn't installed."""

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from videokyc.document import TesseractOcr, classify, extract_fields
from videokyc.document import validators as v
from videokyc.document.classify import DocType
from videokyc.document.ocr import find_tesseract

pytestmark = pytest.mark.skipif(find_tesseract() is None, reason="Tesseract binary not installed")

FONT = ImageFont.load_default(size=30)


def card(lines, skew=0.0, noise=0.0) -> np.ndarray:
    img = Image.new("RGB", (1100, 700), (235, 240, 250))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        d.text((60, 50 + 46 * i), text, fill=(20, 20, 30), font=FONT)
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if skew:
        m = cv2.getRotationMatrix2D((550, 350), skew, 1.0)
        arr = cv2.warpAffine(arr, m, (1100, 700), borderValue=(200, 200, 200))
    if noise:
        arr = np.clip(arr + np.random.default_rng(0).normal(0, noise, arr.shape), 0, 255).astype(np.uint8)
    return arr


@pytest.fixture(scope="module")
def ocr():
    return TesseractOcr()


def read(ocr, img):
    res = ocr.read(img)
    c = classify(res.text)
    return res, c, extract_fields(res.text, c.doc_type)


PAN = ["INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "Permanent Account Number", "ABCPE1234F", "Name",
       "RAHUL KUMAR SHARMA", "Father's Name", "SURESH KUMAR SHARMA", "Date of Birth", "15/08/1990"]


@pytest.mark.parametrize("skew, noise", [(0, 0), (4, 12), (-3, 10)])
def test_pan_card_end_to_end(ocr, skew, noise):
    res, c, f = read(ocr, card(PAN, skew, noise))
    assert c.doc_type is DocType.PAN
    assert (f.id_number, f.name) == ("ABCPE1234F", "Rahul Kumar Sharma")
    assert res.confidence > 60


def test_aadhaar_card_with_verhoeff_number_and_vid(ocr):
    body = "23456789012"
    n = body + v.verhoeff_check_digit(body)
    lines = ["Government of India", "Rahul Kumar Sharma", "DOB: 15/08/1990", "MALE", f"{n[:4]} {n[4:8]} {n[8:]}",
             "Aadhaar - Aam Admi ka Adhikar", "VID: 9123 4567 8901 2345"]
    _, c, f = read(ocr, card(lines, skew=-2, noise=8))
    assert c.doc_type is DocType.AADHAAR
    assert (f.id_number, f.gender, f.name) == (n, "male", "Rahul Kumar Sharma")  # the 16-digit VID is not mistaken for it


def test_driving_licence_and_voter_id(ocr):
    dl = ["Union of India", "Driving Licence", "DL No. KA01 20110012345", "Name : Rahul Kumar Sharma",
          "DOB : 15-08-1990", "Valid Till : 14-08-2031"]
    _, c, f = read(ocr, card(dl))
    assert c.doc_type is DocType.DRIVING_LICENSE and f.id_number == "KA0120110012345"

    voter = ["ELECTION COMMISSION OF INDIA", "ELECTORS PHOTO IDENTITY CARD", "ABC1234567",
             "Elector's Name : Rahul Kumar Sharma", "Sex : MALE", "Date of Birth : 15/08/1990"]
    _, c, f = read(ocr, card(voter))
    assert c.doc_type is DocType.VOTER_ID and f.id_number == "ABC1234567"


def test_non_id_document_is_unknown(ocr):
    _, c, _ = read(ocr, card(["INVOICE #4471", "Total due: Rs 2500", "Thank you for your business"]))
    assert c.doc_type is DocType.UNKNOWN

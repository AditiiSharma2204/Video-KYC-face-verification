"""OCR post-processing that needs no Tesseract: confidence filtering, look-alike repair, deskewing."""

import cv2
import numpy as np
import pytest

from videokyc.document import preprocess
from videokyc.document import validators as v
from videokyc.document.ocr_text import lines_from_data


def data(*words):
    """(text, conf, line) tuples -> pytesseract-style dict; one block/paragraph, x = position in line."""
    return {
        "text": [w[0] for w in words], "conf": [w[1] for w in words],
        "block_num": [1] * len(words), "par_num": [1] * len(words), "line_num": [w[2] for w in words],
        "left": [w[3] if len(w) > 3 else i * 100 for i, w in enumerate(words)],
    }


# -- confidence filtering -------------------------------------------------------------


def test_low_confidence_junk_is_dropped_and_lines_are_rebuilt_in_order():
    d = data(("i", 12, 1, 0), ("Name", 93, 1, 50), ("|", 5, 2, 0), ("Cl", 20, 2, 30), ("RAHUL", 91, 2, 90),
             ("KUMAR", 90, 2, 300))
    text, mean, n = lines_from_data(d, min_conf=40)
    assert text == "Name\nRAHUL KUMAR\n"
    assert n == 3 and mean == pytest.approx((93 + 91 + 90) / 3)


def test_words_are_ordered_by_x_position_not_tesseract_order():
    d = data(("SHARMA", 90, 1, 400), ("RAHUL", 90, 1, 100))
    assert lines_from_data(d, 40)[0] == "RAHUL SHARMA\n"


def test_digit_heavy_tokens_survive_low_confidence_but_words_do_not():
    d = data(("234567890124", 15, 1, 0), ("Rahul", 15, 1, 200), ("15/08/1990", 30, 2, 0), ("ab12", 10, 3, 0))
    text, _, n = lines_from_data(d, 40)
    assert "234567890124" in text and "15/08/1990" in text
    assert "Rahul" not in text and "ab12" not in text  # low-confidence *words* (and short mixed tokens) go
    assert n == 2


def test_empty_result():
    assert lines_from_data(data(), 40) == ("", -1.0, 0)
    assert lines_from_data(data((" ", 90, 1)), 40) == ("", -1.0, 0)


# -- look-alike repair, always gated by a structural validator -------------------------------------


def test_aadhaar_repair_only_accepts_checksum_valid_results():
    good = "23456789012" + v.verhoeff_check_digit("23456789012")  # e.g. 234567890124
    noisy = good[:4] + " " + good[4:8].replace("6", "6") + " " + good[8:]
    noisy_o = noisy.replace("0", "O", 1)  # OCR read a zero as the letter O
    assert v.find_aadhaar_numbers(noisy_o) == [good]

    wrong_digit = noisy_o.replace("5", "6", 1)  # a genuinely misread digit: checksum must reject it
    assert v.find_aadhaar_numbers(wrong_digit) == []


def test_aadhaar_repair_handles_i_l_s_b_confusions():
    good = "23456789012" + v.verhoeff_check_digit("23456789012")
    for digit, glyph in (("1", "I"), ("5", "S"), ("8", "B")):
        if digit in good:
            garbled = f"{good[:4]} {good[4:8]} {good[8:]}".replace(digit, glyph, 1)
            assert v.find_aadhaar_numbers(garbled) == [good], glyph


@pytest.mark.parametrize("ocr, expected", [
    ("ABCPE1234F", "ABCPE1234F"),
    ("ABCPEl234F", "ABCPE1234F"),  # lowercase L read for the digit 1
    ("ABCPEO234F", "ABCPE0234F"),  # letter O where a digit belongs
    ("A8CPE1Z34F", "ABCPE1234F"),  # two look-alikes at once: 8 for B (letter slot), Z for 2 (digit slot)
    ("ABCPE123 4F", None),  # not a 10-character token
    ("ABCZE1234F", None),  # 4th char 'Z' is not a valid holder type: repair cannot invent a valid PAN
])
def test_pan_repair(ocr, expected):
    valid = [n for n in v.find_pan_numbers(f"Permanent Account Number {ocr}") if v.pan_valid(n)]
    assert (valid[0] if valid else None) == expected


def test_dl_repair_letter_o_for_zero():
    assert v.find_dl_numbers("DL No. KAO1 20110012345") == ["KA0120110012345"]
    assert v.find_dl_numbers("DL No. KA01 2O110012345") == ["KA0120110012345"]
    assert v.find_dl_numbers("DL No. KA01 20110012345") == ["KA0120110012345"]
    assert v.find_dl_numbers("DL No. KA01 30110012345") == []  # year 3011 is impossible


# -- deskew ----------------------------------------------------------------------------------------


def text_image(angle=0.0):
    img = np.full((700, 1100, 3), 240, np.uint8)
    for i, line in enumerate(["INCOME TAX DEPARTMENT", "Permanent Account Number", "ABCPE1234F",
                              "RAHUL KUMAR SHARMA", "Date of Birth 15/08/1990"]):
        cv2.putText(img, line, (60, 100 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 3)
    if angle:
        m = cv2.getRotationMatrix2D((550, 350), angle, 1.0)
        img = cv2.warpAffine(img, m, (1100, 700), borderValue=(240, 240, 240))
    return img


@pytest.mark.parametrize("angle", [-6, -3, 3, 8])
def test_skew_estimate_recovers_the_rotation(angle):
    gray = preprocess.to_gray(text_image(angle))
    assert preprocess.estimate_skew(gray) == pytest.approx(-angle, abs=1.0)  # rotate() by this undoes it


def test_straight_and_blank_images_are_left_alone():
    assert preprocess.estimate_skew(preprocess.to_gray(text_image(0))) == 0.0
    assert preprocess.estimate_skew(np.full((300, 300), 255, np.uint8)) == 0.0


def test_variants_return_two_upscaled_gray_images():
    small = np.full((300, 500, 3), 200, np.uint8)
    out = preprocess.variants(small)
    assert len(out) == 2 and all(o.ndim == 2 and o.shape[1] >= preprocess.TARGET_WIDTH for o in out)

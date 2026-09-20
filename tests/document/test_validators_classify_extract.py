from datetime import date

import pytest

from videokyc.document import validators as v
from videokyc.document.classify import DocType, classify
from videokyc.document.extract import extract_fields
from videokyc.privacy import hash_id, mask_id, normalize_id


def make_aadhaar(payload11: str = "23456789012") -> str:
    return payload11 + v.verhoeff_check_digit(payload11)


# -- validators -------------------------------------------------------------


def test_verhoeff_accepts_generated_and_rejects_any_single_digit_error():
    good = make_aadhaar()
    assert v.verhoeff_valid(good) and v.aadhaar_valid(good)
    for i in range(12):  # Verhoeff catches every single-digit substitution
        bad = good[:i] + str((int(good[i]) + 1) % 10) + good[i + 1:]
        assert not v.verhoeff_valid(bad)


def test_verhoeff_catches_adjacent_transposition():
    good = make_aadhaar("34567890123")
    swapped = good[:4] + good[5] + good[4] + good[6:]
    if swapped != good:
        assert not v.verhoeff_valid(swapped)


def test_aadhaar_rules():
    assert not v.aadhaar_valid("0" + make_aadhaar()[1:])  # cannot start with 0/1
    assert not v.aadhaar_valid("1234 5678")  # too short
    spaced = make_aadhaar()
    assert v.aadhaar_valid(f"{spaced[:4]} {spaced[4:8]} {spaced[8:]}")


@pytest.mark.parametrize(
    "fn, good, bad",
    [
        (v.pan_valid, "ABCPE1234F", "ABCZE1234F"),  # 4th letter must be a holder-type code
        (v.passport_valid, "K1234567", "01234567"),
        (v.epic_valid, "ABC1234567", "AB12345678"),
        (v.dl_valid, "KA0120110012345", "K00120110012345"),
    ],
)
def test_format_validators(fn, good, bad):
    assert fn(good) and not fn(bad)


def test_aadhaar_finder_ignores_16_digit_vid_but_keeps_the_real_number():
    assert v.find_aadhaar_numbers("VID: 1234 5678 9012 3456") == []
    assert v.find_aadhaar_numbers("1985\n2345 6789 0123 x") == ["234567890123"]  # number right after a year
    both = "2345 6789 0123\nVID : 9123 4567 8901 2345"
    assert v.find_aadhaar_numbers(both) == ["234567890123"]


# -- privacy ------------------------------------------------------------------------


def test_masking_and_hashing():
    assert mask_id("ABCPE1234F") == "XXXXXX234F"
    assert mask_id("2345 6789 0123") == "XXXXXXXX0123"
    assert normalize_id("abc-12 3") == "ABC123"
    assert hash_id("ABCPE1234F", key=b"k") == hash_id("abcpe 1234 f", key=b"k")  # normalised
    assert hash_id("ABCPE1234F", key=b"k") != hash_id("ABCPE1234F", key=b"other")
    assert "1234" not in hash_id("ABCPE1234F", key=b"k")


# -- classification + extraction on realistic OCR text -----------------------------


AADHAAR_NUM = make_aadhaar()
AADHAAR_FRONT = f"""Government of India
Rahul Kumar Sharma
DOB: 15/08/1990
Male
{AADHAAR_NUM[:4]} {AADHAAR_NUM[4:8]} {AADHAAR_NUM[8:]}
Aadhaar - Aam Admi ka Adhikar
VID: 9123 4567 8901 2345
"""

AADHAAR_BACK = f"""Unique Identification Authority of India
Address:
S/O Suresh Kumar, 12 MG Road, Indiranagar,
Bengaluru, Karnataka - 560038
{AADHAAR_NUM[:4]} {AADHAAR_NUM[4:8]} {AADHAAR_NUM[8:]}
"""

PAN_TEXT = """INCOME TAX DEPARTMENT
GOVT. OF INDIA
Permanent Account Number
ABCPE1234F
Name
RAHUL KUMAR SHARMA
Father's Name
SURESH KUMAR SHARMA
Date of Birth
15/08/1990
Signature
"""

PASSPORT_TEXT = """REPUBLIC OF INDIA
PASSPORT
Type P  Country Code IND  Passport No. K1234567
Surname
SHARMA
Given Name(s)
RAHUL KUMAR
Nationality INDIAN
Date of Expiry 01/01/2030
"""

VOTER_TEXT = """ELECTION COMMISSION OF INDIA
ELECTORS PHOTO IDENTITY CARD
ABC1234567
Elector's Name : Rahul Kumar Sharma
Father's Name : Suresh Kumar Sharma
Sex : MALE
Date of Birth : 15/08/1990
"""

DL_TEXT = """Union of India
Driving Licence
DL No. KA01 20110012345
Name : Rahul Kumar Sharma
S/D/W of : Suresh Kumar Sharma
DOB : 15-08-1990
Valid Till : 14-08-2031
"""


@pytest.mark.parametrize(
    "text, expected",
    [
        (AADHAAR_FRONT, DocType.AADHAAR),
        (AADHAAR_BACK, DocType.AADHAAR),
        (PAN_TEXT, DocType.PAN),
        (PASSPORT_TEXT, DocType.PASSPORT),
        (VOTER_TEXT, DocType.VOTER_ID),
        (DL_TEXT, DocType.DRIVING_LICENSE),
    ],
)
def test_classifies_each_document_type(text, expected):
    c = classify(text)
    assert c.doc_type is expected
    assert c.confidence > 0.3
    assert c.evidence


@pytest.mark.parametrize("text", ["", "hello world this is a grocery list", "Invoice 2345 6789 0123 total 500"])
def test_unrelated_text_is_unknown(text):
    assert classify(text).doc_type is DocType.UNKNOWN


def test_random_twelve_digits_alone_do_not_make_an_aadhaar():
    wrong_check = "23456789012" + str((int(v.verhoeff_check_digit("23456789012")) + 1) % 10)
    assert not v.verhoeff_valid(wrong_check)
    text = f"Order ref {wrong_check[:4]} {wrong_check[4:8]} {wrong_check[8:]} dispatched"
    assert classify(text).doc_type is DocType.UNKNOWN


def test_aadhaar_extraction_front():
    f = extract_fields(AADHAAR_FRONT, DocType.AADHAAR)
    assert f.id_number == AADHAAR_NUM
    assert f.name == "Rahul Kumar Sharma"
    assert f.dob == date(1990, 8, 15)
    assert f.gender == "male"
    assert not f.warnings


def test_aadhaar_extraction_back_address():
    f = extract_fields(AADHAAR_BACK, DocType.AADHAAR)
    assert "12 MG Road" in f.address and f.address.endswith("560038")


def test_aadhaar_bad_checksum_is_flagged_not_hidden():
    bad = AADHAAR_FRONT.replace(AADHAAR_NUM[:4], "9999", 1)
    f = extract_fields(bad, DocType.AADHAAR)
    assert any("checksum" in w for w in f.warnings)


def test_aadhaar_year_of_birth_only():
    text = f"Government of India\nPriya Nair\nYear of Birth : 1985\nFemale\n{AADHAAR_NUM[:4]} {AADHAAR_NUM[4:8]} {AADHAAR_NUM[8:]}"
    f = extract_fields(text, DocType.AADHAAR)
    assert f.year_of_birth == 1985 and f.dob is None and f.gender == "female"
    assert f.age(date(2026, 9, 20)) == 41


def test_pan_extraction():
    f = extract_fields(PAN_TEXT, DocType.PAN)
    assert (f.id_number, f.name, f.father_name, f.dob) == (
        "ABCPE1234F", "Rahul Kumar Sharma", "Suresh Kumar Sharma", date(1990, 8, 15),
    )


def test_voter_and_dl_extraction():
    v_ = extract_fields(VOTER_TEXT, DocType.VOTER_ID)
    assert (v_.id_number, v_.name, v_.gender, v_.dob) == ("ABC1234567", "Rahul Kumar Sharma", "male", date(1990, 8, 15))
    d = extract_fields(DL_TEXT, DocType.DRIVING_LICENSE)
    assert d.id_number == "KA0120110012345" and d.dob == date(1990, 8, 15) and d.name == "Rahul Kumar Sharma"
    # the licence's *validity* date must not be mistaken for the date of birth
    assert d.dob != date(2031, 8, 14)


def test_passport_mrz_icao_specimen():
    mrz = (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10\n"
    )
    f = extract_fields("PASSPORT\n" + mrz, DocType.PASSPORT)
    assert f.id_number == "L898902C3"
    assert f.name == "Anna Maria Eriksson"
    assert f.dob == date(1974, 8, 12) and f.gender == "female"
    assert not any("MRZ" in w for w in f.warnings)


def test_passport_mrz_check_digit_failure_is_reported():
    mrz = (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
        "L898902C37UTO7408122F1204159ZE184226B<<<<<10\n"  # passport-number check digit altered
    )
    f = extract_fields(mrz, DocType.PASSPORT)
    assert any("MRZ" in w for w in f.warnings)


def test_unknown_document_extracts_nothing():
    f = extract_fields("random", DocType.UNKNOWN)
    assert f.id_number is None and f.warnings

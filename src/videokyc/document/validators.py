"""Format and checksum validators for Indian government ID numbers.

Validating the *structure* of an OCR'd number (not just matching a regex) is what
lets the classifier tell a real Aadhaar from any random 12 digits, and lets us
flag OCR misreads.
"""

from __future__ import annotations

import re

# Verhoeff algorithm tables (the checksum UIDAI uses for Aadhaar numbers).
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(payload: str) -> str:
    """Compute the check digit to append to ``payload`` (used to build test fixtures)."""
    c = 0
    for i, ch in enumerate(reversed(payload)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


# The trailing lookahead stops us matching the first 12 digits of a 16-digit VID.
AADHAAR_RE = re.compile(r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b(?!\s?\d)")
VID_RE = re.compile(r"\b\d{4}[ ]?\d{4}[ ]?\d{4}[ ]?\d{4}\b")  # 16-digit Virtual ID printed next to the number
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
PASSPORT_RE = re.compile(r"\b[A-PR-WY][1-9]\d{6}\b")
EPIC_RE = re.compile(r"\b[A-Z]{3}[0-9]{7}\b")
DL_RE = re.compile(r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?(?:19|20)\d{2}[-\s]?\d{7}\b")

# 4th character of a PAN encodes the holder type (P = individual, C = company, ...).
PAN_HOLDER_TYPES = set("ABCFGHLJPT")


def aadhaar_valid(number: str) -> bool:
    digits = re.sub(r"\D", "", number)
    # Aadhaar numbers never start with 0 or 1 and pass the Verhoeff checksum.
    return len(digits) == 12 and digits[0] not in "01" and verhoeff_valid(digits)


def pan_valid(number: str) -> bool:
    n = number.strip().upper()
    return bool(PAN_RE.fullmatch(n)) and n[3] in PAN_HOLDER_TYPES


def passport_valid(number: str) -> bool:
    return bool(PASSPORT_RE.fullmatch(number.strip().upper()))


def epic_valid(number: str) -> bool:
    return bool(EPIC_RE.fullmatch(number.strip().upper()))


def dl_valid(number: str) -> bool:
    return bool(DL_RE.fullmatch(number.strip().upper()))


# -- OCR repair --------------------------------------------------------------------------------
# Tesseract confuses look-alike glyphs. In positions where the format demands a digit (or a
# letter) we can safely translate, then let the structural validator above decide whether the
# repaired value is real. Repairs that fail validation are discarded, never guessed.

_TO_DIGIT = str.maketrans("OoQDIlLSsBZ", "00001115582")  # text is upper-cased first, so 'L' also covers 'l'
_TO_LETTER = str.maketrans("0158", "OISB")
_AADHAAR_LOOSE = re.compile(r"\b([0-9OISBZ]{4})[ ]?([0-9OISBZ]{4})[ ]?([0-9OISBZ]{4})\b(?![ ]?[0-9OISBZ])")
_PAN_TOKEN = re.compile(r"\b[A-Z0-9]{10}\b")
# The year group is loose ([12]xxx) because the 'O' misread can hit the '0' of '20'; dl_valid() then
# re-checks the repaired value against the strict format (19xx/20xx).
_DL_LOOSE = re.compile(r"\b([A-Z]{2})[-\s]?([0-9OIL]{2})[-\s]?([12][0-9OIL]{3})[-\s]?([0-9OIL]{7})\b")


def find_aadhaar_numbers(text: str) -> list[str]:
    """12-digit Aadhaar candidates: strict matches plus checksum-validated OCR repairs.

    The 16-digit VID that Aadhaar prints next to the number is ignored.
    """
    cleaned = VID_RE.sub(" ", text)
    found = ["".join(m.groups()) for m in AADHAAR_RE.finditer(cleaned)]
    for m in _AADHAAR_LOOSE.finditer(cleaned.upper()):
        repaired = "".join(m.groups()).translate(_TO_DIGIT)
        if repaired not in found and aadhaar_valid(repaired):
            found.append(repaired)
    return found


def find_pan_numbers(text: str) -> list[str]:
    upper = text.upper()
    found = [m.group() for m in PAN_RE.finditer(upper)]
    for tok in _PAN_TOKEN.findall(upper):
        letters = (tok[:5] + tok[9]).translate(_TO_LETTER)
        repaired = letters[:5] + tok[5:9].translate(_TO_DIGIT) + letters[5]
        if repaired not in found and pan_valid(repaired):
            found.append(repaired)
    return found


def find_dl_numbers(text: str) -> list[str]:
    found = [re.sub(r"[-\s]", "", m.group()) for m in DL_RE.finditer(text.upper())]
    for m in _DL_LOOSE.finditer(text.upper()):
        state, rto, year, serial = m.groups()
        repaired = state + rto.translate(_TO_DIGIT) + year.translate(_TO_DIGIT) + serial.translate(_TO_DIGIT)
        if repaired not in found and dl_valid(repaired):
            found.append(repaired)
    return found

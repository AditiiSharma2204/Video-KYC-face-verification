"""Turn OCR text into structured ID fields (name, DOB, gender, number, address).

Extraction is layout-aware per document type but tolerant: anything that cannot be
read is left ``None`` and noted in ``warnings`` rather than guessed, so downstream
voice-KYC questions know which answers can actually be verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from . import validators as v
from .classify import DocType

_DATE_RE = re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})\b")
_YOB_RE = re.compile(r"(?:year of birth|yob)\s*[:\-]?\s*(\d{4})", re.I)
_DOB_LABEL = re.compile(r"\b(dob|d\.o\.b|date of birth|birth)\b|जन्म", re.I)
_SKIP_DATE = re.compile(r"issue|valid|expiry|expire|validity|doi|renew", re.I)
_PIN_RE = re.compile(r"\b\d{6}\b")
_GENDER_WORDS = {
    "female": "female", "male": "male", "transgender": "other",
    "महिला": "female", "पुरुष": "male",
}
_NAME_STOP = re.compile(
    r"government|india|authority|aadhaar|aadhar|uidai|dob|birth|male|female|address|"
    r"vid|download|issued|enrol|income|tax|department|election|commission|card|licence|license|"
    r"passport|republic|signature|father|mother|husband|elector",
    re.I,
)


@dataclass
class IdFields:
    doc_type: DocType
    id_number: str | None = None
    name: str | None = None
    dob: date | None = None
    year_of_birth: int | None = None
    gender: str | None = None  # "male" | "female" | "other"
    address: str | None = None
    father_name: str | None = None
    warnings: list[str] = field(default_factory=list)

    def age(self, today: date | None = None) -> int | None:
        today = today or date.today()
        if self.dob:
            return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))
        if self.year_of_birth:
            return today.year - self.year_of_birth
        return None


# -- generic helpers ---------------------------------------------------------


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _parse_date(d: str, m: str, y: str) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def find_dob(lines: list[str]) -> date | None:
    """Prefer a date on/after a 'DOB' label; otherwise the first date that isn't an issue/expiry date."""
    for i, ln in enumerate(lines):
        if _DOB_LABEL.search(ln):
            for cand in (ln, lines[i + 1] if i + 1 < len(lines) else ""):
                m = _DATE_RE.search(cand)
                if m and (dt := _parse_date(*m.groups())):
                    return dt
    for ln in lines:
        if _SKIP_DATE.search(ln):
            continue
        m = _DATE_RE.search(ln)
        if m and (dt := _parse_date(*m.groups())):
            return dt
    return None


def find_gender(text: str) -> str | None:
    lower = text.lower()
    for word, canon in _GENDER_WORDS.items():  # 'female' precedes 'male' so it wins
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lower):
            return canon
    m = re.search(r"(?:sex|gender)\s*[:\-/]?\s*([mf])\b", lower)
    return {"m": "male", "f": "female"}.get(m.group(1)) if m else None


def label_value(lines: list[str], label: re.Pattern[str]) -> str | None:
    """Value after ``label`` on the same line (after ':'), else the next line."""
    for i, ln in enumerate(lines):
        m = label.search(ln)
        if not m:
            continue
        rest = ln[m.end():].lstrip(" :-/.\t")
        rest = re.sub(r"^[^\x00-\x7f]+[\s:/-]*", "", rest)  # drop a Hindi label echo like "नाम"
        if len(re.findall(r"[A-Za-z]", rest)) >= 2:
            return rest.strip()
        if i + 1 < len(lines):
            return lines[i + 1].strip()
    return None


def clean_name(raw: str | None) -> str | None:
    if not raw:
        return None
    s = re.sub(r"[^A-Za-z .'\-]", " ", raw)
    s = re.sub(r"\s+", " ", s).strip(" .-'")
    parts = s.split()
    while len(parts) >= 3 and len(parts[0].strip(".'-")) <= 2 and not parts[0].endswith("."):
        parts.pop(0)  # OCR noise from photo edges, not a real leading initial-less token
    s = " ".join(parts)
    letters = re.sub(r"[^A-Za-z]", "", s)
    if len(letters) < 3 or len(letters) < 0.6 * len(re.sub(r"\s", "", raw)):
        return None  # mostly non-letters => OCR garbage (typically Hindi script read as Latin)
    return s.title()


# -- passport MRZ -----------------------------------------------------------


_MRZ_W = (7, 3, 1)


def _mrz_check(field_: str, check: str) -> bool:
    total = 0
    for i, ch in enumerate(field_):
        val = int(ch) if ch.isdigit() else (ord(ch) - 55 if ch.isalpha() else 0)
        total += val * _MRZ_W[i % 3]
    return check.isdigit() and total % 10 == int(check)


def _mrz_birth_date(yymmdd: str) -> date | None:
    """MRZ dates have 2-digit years: a birth year later than 'now' must be last century."""
    if not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    year = (2000 if yy <= date.today().year % 100 else 1900) + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def parse_mrz(lines: list[str]) -> IdFields | None:
    cands = [re.sub(r"\s", "", ln.upper()) for ln in lines]
    cands = [c for c in cands if len(c) >= 40 and re.fullmatch(r"[A-Z0-9<]+", c)]
    for a, b in zip(cands, cands[1:], strict=False):
        if not a.startswith("P<"):
            continue
        a, b = a[:44].ljust(44, "<"), b[:44].ljust(44, "<")
        surname, _, given = a[5:].partition("<<")
        f = IdFields(DocType.PASSPORT)
        f.name = clean_name(f"{given.replace('<', ' ')} {surname.replace('<', ' ')}")
        number = b[:9].replace("<", "")
        f.id_number = number or None
        dob_raw, sex = b[13:19], b[20]
        f.dob = _mrz_birth_date(dob_raw)
        f.gender = {"M": "male", "F": "female"}.get(sex, "other" if sex == "X" else None)
        if not (_mrz_check(b[:9], b[9]) and _mrz_check(dob_raw, b[19])):
            f.warnings.append("MRZ check digits failed - OCR may have misread the machine-readable zone")
        return f
    return None


# -- per-document extractors -------------------------------------------------


def _pick_number(candidates: list[str], valid, doc: IdFields, label: str) -> None:
    good = [n for n in candidates if valid(n)]
    if good:
        doc.id_number = re.sub(r"[-\s]", "", good[0])
    elif candidates:
        doc.id_number = re.sub(r"[-\s]", "", candidates[0])
        doc.warnings.append(f"{label} number failed validation - check for OCR errors")


def _strict(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [m.group() for m in pattern.finditer(text.upper())]


def _extract_aadhaar(text: str, lines: list[str]) -> IdFields:
    f = IdFields(DocType.AADHAAR)
    nums = v.find_aadhaar_numbers(text)
    valid = [n for n in nums if v.aadhaar_valid(n)]
    if valid:
        f.id_number = valid[0]
    elif nums:
        f.id_number = nums[0]
        f.warnings.append("Aadhaar number failed the Verhoeff checksum - possible OCR error")

    f.dob = find_dob(lines)
    if not f.dob and (m := _YOB_RE.search(text)):
        f.year_of_birth = int(m.group(1))
    f.gender = find_gender(text)

    # The English name sits just above the DOB line on the front of the card.
    # Anchor on the DOB label, or on the date itself when OCR dropped the label.
    anchor = next(
        (i for i, ln in enumerate(lines) if _DOB_LABEL.search(ln) or _YOB_RE.search(ln) or _DATE_RE.search(ln)),
        None,
    )
    if anchor is not None:
        for ln in reversed(lines[:anchor]):
            name = clean_name(ln)
            if name and not _NAME_STOP.search(name) and len(name.split()) >= 2:
                f.name = name
                break

    addr = label_value(lines, re.compile(r"^[^A-Za-z]*(?:[A-Za-z]{1,2}[^A-Za-z]+)?address\s*[:\-]?", re.I))
    if addr is not None:
        idx = next((i for i, ln in enumerate(lines) if re.match(r"^address\b", ln, re.I)), None)
        parts: list[str] = []
        for ln in lines[idx:] if idx is not None else []:
            parts.append(re.sub(r"^address\s*[:\-]?\s*", "", ln, flags=re.I))
            if _PIN_RE.search(ln):
                break
        f.address = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip() or None
    return f


def _extract_pan(text: str, lines: list[str]) -> IdFields:
    f = IdFields(DocType.PAN)
    _pick_number(v.find_pan_numbers(text), v.pan_valid, f, "PAN")
    f.name = clean_name(label_value(lines, re.compile(r"^name\b", re.I)))
    f.father_name = clean_name(label_value(lines, re.compile(r"father'?s?\s*name", re.I)))
    f.dob = find_dob(lines)
    f.gender = find_gender(text)
    return f


def _extract_passport(text: str, lines: list[str]) -> IdFields:
    mrz = parse_mrz(lines)
    f = mrz or IdFields(DocType.PASSPORT)
    if not f.id_number:
        _pick_number(_strict(text, v.PASSPORT_RE), v.passport_valid, f, "Passport")
    if not f.name:
        given = clean_name(label_value(lines, re.compile(r"given name\(?s?\)?", re.I)))
        sur = clean_name(label_value(lines, re.compile(r"^[^A-Za-z]*(?:[A-Za-z]{1,2}[^A-Za-z]+)?surname", re.I)))
        f.name = " ".join(p for p in (given, sur) if p) or None
    f.dob = f.dob or find_dob(lines)
    f.gender = f.gender or find_gender(text)
    f.address = label_value(lines, re.compile(r"^[^A-Za-z]*(?:[A-Za-z]{1,2}[^A-Za-z]+)?address\s*[:\-]?", re.I))
    return f


def _extract_voter(text: str, lines: list[str]) -> IdFields:
    f = IdFields(DocType.VOTER_ID)
    _pick_number(_strict(text, v.EPIC_RE), v.epic_valid, f, "EPIC")
    f.name = clean_name(label_value(lines, re.compile(r"(?:elector'?s?\s*)?name\s*[:\-]", re.I)))
    f.father_name = clean_name(
        label_value(lines, re.compile(r"(?:father|husband|relation)'?s?\s*name", re.I))
    )
    f.dob = find_dob(lines)
    if not f.dob and (m := re.search(r"age\s*(?:as on.*?)?[:\-]?\s*(\d{2})\b", text, re.I)):
        f.year_of_birth = date.today().year - int(m.group(1))
    f.gender = find_gender(text)
    f.address = label_value(lines, re.compile(r"^[^A-Za-z]*(?:[A-Za-z]{1,2}[^A-Za-z]+)?address\s*[:\-]?", re.I))
    return f


def _extract_dl(text: str, lines: list[str]) -> IdFields:
    f = IdFields(DocType.DRIVING_LICENSE)
    _pick_number(v.find_dl_numbers(text), v.dl_valid, f, "Driving licence")
    f.name = clean_name(label_value(lines, re.compile(r"^[^A-Za-z]*(?:[A-Za-z]{1,2}[^A-Za-z]+)?name\s*[:\-]", re.I)))
    f.father_name = clean_name(label_value(lines, re.compile(r"s/?d/?w\s*(?:of)?", re.I)))
    f.dob = find_dob(lines)
    f.gender = find_gender(text)
    f.address = label_value(lines, re.compile(r"^[^A-Za-z]*(?:[A-Za-z]{1,2}[^A-Za-z]+)?address\s*[:\-]?", re.I))
    return f


_EXTRACTORS = {
    DocType.AADHAAR: _extract_aadhaar,
    DocType.PAN: _extract_pan,
    DocType.PASSPORT: _extract_passport,
    DocType.VOTER_ID: _extract_voter,
    DocType.DRIVING_LICENSE: _extract_dl,
}


def extract_fields(text: str, doc_type: DocType) -> IdFields:
    if doc_type is DocType.UNKNOWN:
        return IdFields(DocType.UNKNOWN, warnings=["Document type not recognised"])
    fields = _EXTRACTORS[doc_type](text, _lines(text))
    for attr, label in (("id_number", "ID number"), ("name", "name"), ("dob", "date of birth")):
        if getattr(fields, attr) is None and not (attr == "dob" and fields.year_of_birth):
            fields.warnings.append(f"Could not read {label}")
    return fields

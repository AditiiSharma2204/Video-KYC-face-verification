"""Rule-based document-type classification from OCR text.

Each candidate type earns points from (a) characteristic keywords and (b) an
ID-number pattern that also passes its structural check (Verhoeff for Aadhaar,
holder-type letter for PAN, ...). A rules engine is deliberately chosen over an
ML classifier: it is explainable (every decision lists its evidence), needs no
labelled data, and is robust to OCR noise because number formats are strong signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import validators as v


class DocType(str, Enum):
    AADHAAR = "aadhaar"
    PAN = "pan"
    PASSPORT = "passport"
    VOTER_ID = "voter_id"
    DRIVING_LICENSE = "driving_license"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Classification:
    doc_type: DocType
    confidence: float  # 0-1
    scores: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


KEYWORDS: dict[DocType, list[str]] = {
    DocType.AADHAAR: [
        "aadhaar", "aadhar", "uidai", "unique identification authority", "आधार",
        "mera aadhaar", "enrolment", "enrollment no", "vid :", "vid:",
    ],
    DocType.PAN: [
        "income tax department", "permanent account number", "आयकर विभाग", "pan card",
        "income tax", "signature",
    ],
    DocType.PASSPORT: [
        "passport", "republic of india", "p<ind", "place of issue", "date of expiry",
        "nationality", "type/type", "country code", "surname", "given name",
    ],
    DocType.VOTER_ID: [
        "election commission of india", "elector", "electors photo identity card",
        "identity card", "epic no", "निर्वाचन", "voter",
    ],
    DocType.DRIVING_LICENSE: [
        "driving licence", "driving license", "dl no", "union of india", "transport",
        "valid till", "date of first issue", "cov", "authorisation to drive",
    ],
}

# Distinctive keywords count double; generic ones (e.g. "signature") count less.
STRONG = {
    DocType.AADHAAR: {"aadhaar", "aadhar", "uidai", "unique identification authority", "आधार"},
    DocType.PAN: {"permanent account number", "income tax department", "आयकर विभाग"},
    DocType.PASSPORT: {"passport", "p<ind"},
    DocType.VOTER_ID: {"election commission of india", "electors photo identity card"},
    DocType.DRIVING_LICENSE: {"driving licence", "driving license", "authorisation to drive"},
}

MIN_SCORE = 2.5


def find_id_numbers(text: str) -> dict[DocType, list[str]]:
    """All ID-number-shaped tokens in ``text``, keyed by the type they would belong to."""
    upper = text.upper()
    found: dict[DocType, list[str]] = {t: [] for t in DocType if t is not DocType.UNKNOWN}
    found[DocType.AADHAAR] = v.find_aadhaar_numbers(text)
    found[DocType.PAN] = v.find_pan_numbers(text)
    found[DocType.PASSPORT] = [m.group() for m in v.PASSPORT_RE.finditer(upper)]
    found[DocType.VOTER_ID] = [m.group() for m in v.EPIC_RE.finditer(upper)]
    found[DocType.DRIVING_LICENSE] = v.find_dl_numbers(text)
    return found


def classify(text: str) -> Classification:
    lower = text.lower()
    numbers = find_id_numbers(text)
    scores: dict[DocType, float] = {}
    evidence: dict[DocType, list[str]] = {}

    for doc_type, words in KEYWORDS.items():
        pts, ev = 0.0, []
        for w in words:
            if w in lower:
                weight = 2.0 if w in STRONG.get(doc_type, ()) else 1.0
                pts += weight
                ev.append(f"keyword '{w}'")
        scores[doc_type], evidence[doc_type] = pts, ev

    def bonus(doc_type: DocType, valid, strong: float, weak: float) -> None:
        for n in numbers[doc_type]:
            if valid(n):
                scores[doc_type] += strong
                evidence[doc_type].append(f"valid {doc_type.value} number pattern")
                return
        if numbers[doc_type] and weak:
            scores[doc_type] += weak
            evidence[doc_type].append(f"{doc_type.value}-like number (unvalidated)")

    bonus(DocType.AADHAAR, v.aadhaar_valid, 4.0, 0.5)  # random 12 digits are common; only checksum counts
    bonus(DocType.PAN, v.pan_valid, 4.0, 1.0)
    bonus(DocType.PASSPORT, v.passport_valid, 1.5, 0.0)  # pattern is loose, so keywords carry more weight
    bonus(DocType.VOTER_ID, v.epic_valid, 1.5, 0.0)
    bonus(DocType.DRIVING_LICENSE, v.dl_valid, 3.0, 0.0)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top, top_score), (_, second_score) = ranked[0], ranked[1]
    if top_score < MIN_SCORE:
        return Classification(DocType.UNKNOWN, 0.0, {t.value: s for t, s in scores.items()})

    margin = top_score / (top_score + second_score + 1e-9)  # how far ahead of the runner-up
    evidence_strength = min(1.0, top_score / 6.0)  # how much absolute evidence backs it
    return Classification(
        top, round(margin * evidence_strength, 3),
        {t.value: s for t, s in scores.items()}, evidence[top],
    )

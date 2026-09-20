"""The 10 Voice-KYC questions and how each answer is judged.

Questions fall into three kinds:

* **verify**   - the answer is checked against a field read from the ID (age, name,
  DOB, number, gender, address). If the field could not be read from the ID the
  answer is only *recorded*, never failed: we don't reject someone because our OCR
  missed a line.
* **declare**  - self-declarations that nothing on the ID can confirm (previous
  updates, bank linking). Recorded for the audit trail.
* **consent**  - must be an explicit "yes".

Nothing here stores the transcript; ``AnswerResult`` keeps only verdict + score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import Enum

from ..document.extract import IdFields
from ..privacy import normalize_id
from . import parsing as p


class Verdict(str, Enum):
    PASS = "pass"  # noqa: S105 - verdict label, not a credential
    FAIL = "fail"
    RECORDED = "recorded"  # accepted but not verifiable against the ID
    UNCLEAR = "unclear"  # couldn't interpret the answer: ask again


@dataclass(frozen=True)
class AnswerResult:
    question_id: str
    verdict: Verdict
    score: float | None = None  # similarity 0-1 where meaningful
    detail: str = ""  # human-readable reason; never contains the ID value itself


@dataclass(frozen=True)
class VoiceConfig:
    name_min_similarity: float = 0.80
    address_min_overlap: float = 0.50
    number_max_edits: int = 1  # STT slips on long digit strings; 1 edit still needs the real document
    max_attempts: int = 2
    max_failed_answers: int = 0  # verify-type failures tolerated before rejecting the session


@dataclass(frozen=True)
class Context:
    fields: IdFields
    config: VoiceConfig
    today: date


Evaluator = Callable[[str, Context], AnswerResult]


@dataclass(frozen=True)
class Question:
    id: str
    prompt: str
    kind: str  # "verify" | "declare" | "consent"
    evaluate: Evaluator


def _unclear(qid: str, why: str) -> AnswerResult:
    return AnswerResult(qid, Verdict.UNCLEAR, detail=why)


def _age(text: str, ctx: Context) -> AnswerResult:
    said = p.extract_int(text)
    if said is None or not 0 < said < 120:
        return _unclear("age", "Could not hear an age")
    expected = ctx.fields.age(ctx.today)
    if expected is None:
        return AnswerResult("age", Verdict.RECORDED, detail="Age not readable from ID")
    ok = said == expected
    return AnswerResult("age", Verdict.PASS if ok else Verdict.FAIL, detail="" if ok else "Age does not match ID")


def _citizenship(text: str, ctx: Context) -> AnswerResult:
    said = p.parse_yes_no(text)
    if said is None:
        return _unclear("citizenship", "Please answer yes or no")
    return AnswerResult(
        "citizenship", Verdict.PASS if said else Verdict.FAIL, detail="" if said else "Not an Indian citizen"
    )


def _name(text: str, ctx: Context) -> AnswerResult:
    if not p.tokenize(text):
        return _unclear("name", "Could not hear a name")
    if not ctx.fields.name:
        return AnswerResult("name", Verdict.RECORDED, detail="Name not readable from ID")
    score = p.name_similarity(text, ctx.fields.name)
    ok = score >= ctx.config.name_min_similarity
    return AnswerResult("name", Verdict.PASS if ok else Verdict.FAIL, round(score, 3),
                        "" if ok else "Name does not match ID")


def _dob(text: str, ctx: Context) -> AnswerResult:
    said = p.parse_date(text)
    if said is None:
        return _unclear("dob", "Could not understand the date")
    f = ctx.fields
    if f.dob:
        ok = said == f.dob
    elif f.year_of_birth:
        ok = said.year == f.year_of_birth
    else:
        return AnswerResult("dob", Verdict.RECORDED, detail="Date of birth not readable from ID")
    return AnswerResult(
        "dob", Verdict.PASS if ok else Verdict.FAIL, detail="" if ok else "Date of birth does not match ID"
    )


def _doc_number(text: str, ctx: Context) -> AnswerResult:
    said = p.spoken_id(text)
    if len(said) < 4:
        return _unclear("doc_number", "Could not hear the document number")
    if not ctx.fields.id_number:
        return AnswerResult("doc_number", Verdict.RECORDED, detail="Number not readable from ID")
    expected = normalize_id(ctx.fields.id_number)
    edits = p.edit_distance(said, expected)
    ok = edits <= ctx.config.number_max_edits and abs(len(said) - len(expected)) <= ctx.config.number_max_edits
    score = round(1 - edits / max(len(expected), 1), 3)
    return AnswerResult("doc_number", Verdict.PASS if ok else Verdict.FAIL, score,
                        "" if ok else "Document number does not match ID")


def _gender(text: str, ctx: Context) -> AnswerResult:
    said = p.parse_gender(text)
    if said is None:
        return _unclear("gender", "Could not understand the answer")
    if not ctx.fields.gender:
        return AnswerResult("gender", Verdict.RECORDED, detail="Gender not readable from ID")
    ok = said == ctx.fields.gender
    return AnswerResult("gender", Verdict.PASS if ok else Verdict.FAIL, detail="" if ok else "Gender does not match ID")


def _address(text: str, ctx: Context) -> AnswerResult:
    if len(p.tokenize(text)) < 3:
        return _unclear("address", "Please state your full address")
    if not ctx.fields.address:
        return AnswerResult("address", Verdict.RECORDED, detail="Address not readable from ID")
    score = p.address_overlap(text, ctx.fields.address)
    ok = score >= ctx.config.address_min_overlap
    return AnswerResult("address", Verdict.PASS if ok else Verdict.FAIL, round(score, 3),
                        "" if ok else "Address does not match ID")


def _declaration(qid: str) -> Evaluator:
    def evaluate(text: str, ctx: Context) -> AnswerResult:
        said = p.parse_yes_no(text)
        if said is None:
            return _unclear(qid, "Please answer yes or no")
        return AnswerResult(qid, Verdict.RECORDED, detail="yes" if said else "no")

    return evaluate


def _consent(text: str, ctx: Context) -> AnswerResult:
    said = p.parse_yes_no(text)
    if said is None:
        return _unclear("consent", "Please answer yes or no")
    return AnswerResult("consent", Verdict.PASS if said else Verdict.FAIL, detail="" if said else "Consent refused")


QUESTIONS: tuple[Question, ...] = (
    Question("age", "What is your age?", "verify", _age),
    Question("citizenship", "Are you a citizen of India?", "verify", _citizenship),
    Question("name", "Please state your full name as on your ID.", "verify", _name),
    Question("dob", "What is your date of birth?", "verify", _dob),
    Question("doc_number", "Please read out your ID document number.", "verify", _doc_number),
    Question("gender", "What is your gender?", "verify", _gender),
    Question("address", "Please state your address as on your ID.", "verify", _address),
    Question("previous_updates", "Have you updated your details on this ID before?", "declare",
             _declaration("previous_updates")),
    Question("bank_linking", "Is this ID linked to your bank account?", "declare", _declaration("bank_linking")),
    Question("consent", "Do you consent to this video KYC verification and to the storage of its outcome?",
             "consent", _consent),
)

from datetime import date

import pytest

from videokyc.document.classify import DocType
from videokyc.document.extract import IdFields
from videokyc.voice.flow import VoiceFlow
from videokyc.voice.questions import QUESTIONS, Context, Verdict, VoiceConfig

TODAY = date(2026, 9, 20)  # after the 1990-08-15 birthday => age 36
FIELDS = IdFields(
    DocType.PAN, id_number="ABCPE1234F", name="Rahul Kumar Sharma", dob=date(1990, 8, 15), gender="male",
    address="S/O Suresh Kumar, 12 MG Road, Indiranagar, Bengaluru, Karnataka - 560038",
)
Q = {q.id: q for q in QUESTIONS}


def ask(qid, text, fields=FIELDS, **cfg):
    return Q[qid].evaluate(text, Context(fields, VoiceConfig(**cfg), TODAY))


def test_there_are_ten_questions_in_the_designed_order():
    assert [q.id for q in QUESTIONS] == [
        "age", "citizenship", "name", "dob", "doc_number", "gender", "address",
        "previous_updates", "bank_linking", "consent",
    ]


@pytest.mark.parametrize(
    "qid, text, verdict",
    [
        ("age", "thirty six", Verdict.PASS), ("age", "I am 36", Verdict.PASS),
        ("age", "thirty five", Verdict.FAIL), ("age", "banana", Verdict.UNCLEAR),
        ("citizenship", "Yes I am Indian", Verdict.PASS), ("citizenship", "no", Verdict.FAIL),
        ("citizenship", "hmm", Verdict.UNCLEAR),
        ("name", "Rahul Kumar Sharma", Verdict.PASS), ("name", "Sharma Rahul Kumar", Verdict.PASS),
        ("name", "Amit Verma", Verdict.FAIL), ("name", "", Verdict.UNCLEAR),
        ("dob", "15th August 1990", Verdict.PASS), ("dob", "15/08/1990", Verdict.PASS),
        ("dob", "16th August 1990", Verdict.FAIL), ("dob", "sometime in august", Verdict.UNCLEAR),
        ("doc_number", "A B C P E 1 2 3 4 F", Verdict.PASS),
        ("doc_number", "alpha bravo charlie papa echo one two three four foxtrot", Verdict.PASS),
        ("doc_number", "A B C P E 1 2 3 5 F", Verdict.PASS),  # one STT slip tolerated
        ("doc_number", "A B C P E 9 9 9 9 F", Verdict.FAIL), ("doc_number", "um", Verdict.UNCLEAR),
        ("gender", "male", Verdict.PASS), ("gender", "female", Verdict.FAIL), ("gender", "purple", Verdict.UNCLEAR),
        ("address", "12 MG Road Indiranagar Bengaluru Karnataka 560038", Verdict.PASS),
        ("address", "I live in Delhi near the metro", Verdict.FAIL), ("address", "Delhi", Verdict.UNCLEAR),
        ("previous_updates", "yes I updated my address", Verdict.RECORDED),
        ("previous_updates", "no", Verdict.RECORDED), ("previous_updates", "perhaps", Verdict.UNCLEAR),
        ("bank_linking", "yes", Verdict.RECORDED),
        ("consent", "yes I consent", Verdict.PASS), ("consent", "no", Verdict.FAIL),
    ],
)
def test_question_verdicts(qid, text, verdict):
    assert ask(qid, text).verdict is verdict


def test_age_uses_birthday_not_just_year():
    before_birthday = date(2026, 8, 1)  # 35 until 15 Aug
    r = Q["age"].evaluate("thirty five", Context(FIELDS, VoiceConfig(), before_birthday))
    assert r.verdict is Verdict.PASS


def test_unreadable_id_fields_are_recorded_not_failed():
    blank = IdFields(DocType.PAN)
    for qid, text in [("age", "36"), ("name", "Rahul Sharma"), ("dob", "15 08 1990"),
                      ("doc_number", "1 2 3 4 5 6"), ("gender", "male"),
                      ("address", "12 MG Road Bengaluru")]:
        assert ask(qid, text, fields=blank).verdict is Verdict.RECORDED, qid


def test_year_of_birth_only_ids():
    yob = IdFields(DocType.AADHAAR, year_of_birth=1990)
    assert ask("dob", "15 August 1990", fields=yob).verdict is Verdict.PASS
    assert ask("dob", "15 August 1991", fields=yob).verdict is Verdict.FAIL
    assert ask("age", "36", fields=yob).verdict is Verdict.PASS


def test_document_number_error_budget_is_configurable():
    assert ask("doc_number", "A B C P E 1 2 3 5 F", number_max_edits=0).verdict is Verdict.FAIL


def test_results_never_echo_the_id_value():
    r = ask("doc_number", "A B C P E 9 9 9 9 F")
    assert "ABCPE1234F" not in r.detail and "Sharma" not in ask("name", "Amit Verma").detail


# -- the dialogue ---------------------------------------------------------------------------


def perfect_answers():
    return [
        "thirty six", "yes", "Rahul Kumar Sharma", "15th August 1990", "A B C P E 1 2 3 4 F", "male",
        "12 MG Road Indiranagar Bengaluru Karnataka 560038", "no", "yes", "I consent",
    ]


def run(flow, answers):
    for a in answers:
        if flow.finished:
            break
        flow.answer(a)
    return flow


def test_happy_path_passes():
    flow = run(VoiceFlow(FIELDS, today=TODAY), perfect_answers())
    assert flow.finished and flow.passed and not flow.failures
    assert [r.question_id for r in flow.results][-1] == "consent"


def test_unclear_answer_gets_a_retry_then_advances():
    flow = VoiceFlow(FIELDS, today=TODAY)
    r = flow.answer("mumble")
    assert r.verdict is Verdict.UNCLEAR and flow.retry_pending and flow.current.id == "age"
    assert flow.attempts_left == 1
    r = flow.answer("thirty six")
    assert r.verdict is Verdict.PASS and not flow.retry_pending and flow.current.id == "citizenship"


def test_wrong_answer_can_be_corrected_on_retry():
    flow = VoiceFlow(FIELDS, today=TODAY)
    assert flow.answer("thirty").verdict is Verdict.FAIL and flow.retry_pending
    assert flow.answer("thirty six").verdict is Verdict.PASS and not flow.failures


def test_two_wrong_answers_record_a_failure_and_fail_the_flow():
    answers = perfect_answers()
    answers.insert(0, "twenty")  # wrong age on attempt 1 ...
    answers[1] = "twenty one"  # ... and attempt 2 => recorded FAIL, then the rest are fine
    flow = run(VoiceFlow(FIELDS, today=TODAY), answers)
    assert [f.question_id for f in flow.failures] == ["age"]
    assert flow.finished and not flow.passed


def test_persistently_unclear_becomes_a_failure():
    flow = VoiceFlow(FIELDS, today=TODAY)
    flow.answer("um")
    r = flow.answer("uh")
    assert r.verdict is Verdict.FAIL and "No usable answer" in r.detail


def test_consent_is_mandatory_even_if_everything_else_passes():
    answers = perfect_answers()[:-1] + ["no", "no"]
    flow = run(VoiceFlow(FIELDS, today=TODAY), answers)
    assert flow.finished and not flow.passed


def test_tolerating_failures_is_configurable():
    answers = perfect_answers()
    answers[5] = "female"  # gender wrong on both attempts
    answers.insert(6, "female")
    flow = run(VoiceFlow(FIELDS, VoiceConfig(max_failed_answers=1), today=TODAY), answers)
    assert len(flow.failures) == 1 and flow.passed


def test_answering_after_finish_raises():
    flow = run(VoiceFlow(FIELDS, today=TODAY), perfect_answers())
    with pytest.raises(RuntimeError):
        flow.answer("hello")

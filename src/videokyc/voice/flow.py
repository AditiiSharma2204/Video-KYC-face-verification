"""Stateful Voice-KYC dialogue: asks the questions in order, allows retries, and
decides whether the voice stage passed."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..document.extract import IdFields
from .questions import QUESTIONS, AnswerResult, Context, Question, Verdict, VoiceConfig


@dataclass
class VoiceFlow:
    fields: IdFields
    config: VoiceConfig = field(default_factory=VoiceConfig)
    questions: tuple[Question, ...] = QUESTIONS
    today: date = field(default_factory=date.today)
    index: int = 0
    attempts: int = 0
    results: list[AnswerResult] = field(default_factory=list)
    retry_pending: bool = False  # True right after a non-final FAIL/UNCLEAR: same question is asked again

    @property
    def finished(self) -> bool:
        return self.index >= len(self.questions)

    @property
    def current(self) -> Question | None:
        return None if self.finished else self.questions[self.index]

    @property
    def attempts_left(self) -> int:
        return max(0, self.config.max_attempts - self.attempts)

    def answer(self, transcript: str) -> AnswerResult:
        """Judge ``transcript`` against the current question.

        UNCLEAR and FAIL answers earn a retry until ``max_attempts`` is used up; the final
        result for the question is then recorded and the flow moves on.
        """
        q = self.current
        if q is None:
            raise RuntimeError("Voice flow already finished")
        ctx = Context(self.fields, self.config, self.today)
        result = q.evaluate(transcript, ctx)
        self.attempts += 1

        retryable = result.verdict in (Verdict.UNCLEAR, Verdict.FAIL)
        self.retry_pending = retryable and self.attempts < self.config.max_attempts
        if self.retry_pending:
            return result  # stay on this question; caller shows the retry prompt
        if result.verdict is Verdict.UNCLEAR:  # out of attempts and still unintelligible
            result = AnswerResult(result.question_id, Verdict.FAIL, detail="No usable answer after retries")
        self.results.append(result)
        self.index += 1
        self.attempts = 0
        return result

    @property
    def failures(self) -> list[AnswerResult]:
        return [r for r in self.results if r.verdict is Verdict.FAIL]

    @property
    def passed(self) -> bool:
        """All questions answered, consent given, and verify-type failures within tolerance."""
        if not self.finished:
            return False
        consent_ok = any(r.question_id == "consent" and r.verdict is Verdict.PASS for r in self.results)
        return consent_ok and len(self.failures) <= self.config.max_failed_answers

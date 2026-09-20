"""Turn speech-to-text transcripts into comparable values (numbers, dates, IDs, yes/no).

Speech recognisers write the same answer many ways ("thirty two", "32", "thirty-two
years"), so every question normalises its transcript here before comparing it with
the value read from the ID. All functions are pure and return ``None`` when the
transcript can't be interpreted, which the flow treats as "please repeat".
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher

_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]
    )
}
_MONTHS.update({m[:3]: n for m, n in list(_MONTHS.items())} | {"sept": 9})

_NATO = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F",
    "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "kilo": "K", "lima": "L", "mike": "M",
    "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S",
    "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X", "yankee": "Y", "zulu": "Z",
}
_MULTIPLIERS = {"double": 2, "triple": 3}

_YES = {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "affirmative", "absolutely", "definitely",
        "agree", "consent", "haan", "han", "ji", "okay", "ok", "certainly", "true"}
_NO = {"no", "nope", "nah", "never", "nahi", "negative", "incorrect", "false", "don't", "dont", "not", "disagree"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))


_tokens = tokenize


# -- numbers ----------------------------------------------------------------


def words_to_number(text: str) -> int | None:
    """'thirty two' / 'one hundred and five' / 'twenty-five' -> int. None if no number words."""
    total, current, seen = 0, 0, False
    for tok in _tokens(text):
        if tok in _UNITS:
            current += _UNITS[tok]
            seen = True
        elif tok in _TENS:
            current += _TENS[tok]
            seen = True
        elif tok == "hundred":
            current = max(current, 1) * 100
            seen = True
        elif tok == "thousand":
            total += max(current, 1) * 1000
            current = 0
            seen = True
        elif tok == "and" and seen:
            continue
        elif seen:
            break  # first non-number word after the number ends it ("thirty two years old")
    return total + current if seen else None


def extract_int(text: str) -> int | None:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else words_to_number(text)


# -- dates ------------------------------------------------------------------


def _spoken_year(tokens: list[str], start: int) -> tuple[int | None, int]:
    """Parse a year starting at tokens[start]; returns (year, tokens_consumed).

    Handles '1990', 'nineteen ninety', 'nineteen oh five', 'two thousand five', 'two thousand'.
    """
    tok = tokens[start] if start < len(tokens) else ""
    if re.fullmatch(r"\d{4}", tok):
        return int(tok), 1
    rest = tokens[start:]
    if len(rest) >= 2 and rest[0] in ("nineteen", "twenty") and (rest[1] in _TENS or rest[1] in _UNITS):
        century = 1900 if rest[0] == "nineteen" else 2000
        if rest[1] in _TENS:
            yy = _TENS[rest[1]]
            if len(rest) >= 3 and rest[2] in _UNITS and _UNITS[rest[2]] < 10:
                return century + yy + _UNITS[rest[2]], 3
            return century + yy, 2
        if rest[1] in ("oh", "zero") and len(rest) >= 3 and rest[2] in _UNITS and _UNITS[rest[2]] < 10:
            return century + _UNITS[rest[2]], 3
        if 10 <= _UNITS[rest[1]] <= 19 and rest[0] == "twenty":
            return century + _UNITS[rest[1]], 2
    if rest[:2] == ["two", "thousand"]:
        if len(rest) >= 3 and (rest[2] in _UNITS or rest[2] in _TENS):
            extra = words_to_number(" ".join(rest[2:4]))
            return 2000 + (extra or 0), 2 + (2 if extra and extra >= 20 and len(rest) >= 4 else 1)
        return 2000, 2
    return None, 0


def parse_date(text: str) -> date | None:
    """Parse a spoken/written date. Day-first is assumed for all-numeric forms (Indian convention)."""
    tokens = _tokens(text)
    if not tokens:
        return None

    # numeric: 15/08/1990, 15 08 1990, 15-8-90 is rejected (2-digit years are ambiguous)
    m = re.search(r"(\d{1,2})\s*[/\-. ]\s*(\d{1,2})\s*[/\-. ]\s*(\d{4})", text)
    if m:
        return _mk(int(m.group(3)), int(m.group(2)), int(m.group(1)))

    found = _find_month(tokens)
    if found is None:
        return None
    month_idx, month = found

    # day: a digit token (optionally with st/nd/rd/th), an ordinal word, or a number word
    day = None
    for i, t in enumerate(tokens):
        if i == month_idx:
            continue
        mm = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", t)
        if mm and not re.fullmatch(r"\d{4}", t):
            day = int(mm.group(1))
            break
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t in _TENS and nxt in _ORDINALS and _ORDINALS[nxt] < 10:  # 'twenty first', 'thirty first'
            day = _TENS[t] + _ORDINALS[nxt]
            break
        if t in _ORDINALS:
            day = _ORDINALS[t]
            break
        if t in _TENS or t in _UNITS:
            n = words_to_number(" ".join(tokens[i:i + 2]))
            if n and 1 <= n <= 31:
                day = n
                break
    if day is None:
        return None

    year = None
    for i in range(len(tokens)):
        y, used = _spoken_year(tokens, i)
        if y and used and (i != month_idx):
            year = y
            break
    if year is None:
        return None
    return _mk(year, month, day)


def _find_month(tokens: list[str]) -> tuple[int, int] | None:
    """(index, month number) of the first month word. 'may' only counts when a year is present,
    so the verb in 'may I say...' is not mistaken for a month."""
    has_year = any(re.fullmatch(r"\d{4}", t) or t in ("nineteen", "twenty", "thousand") for t in tokens)
    for i, t in enumerate(tokens):
        if t in _MONTHS and (t != "may" or has_year):
            return i, _MONTHS[t]
    return None


def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


# -- identifiers --------------------------------------------------------------


def spoken_id(text: str) -> str:
    """'A B C D E one two three four F' / 'double five' / 'alpha bravo 1234' -> 'ABCDE1234F'-style string."""
    out: list[str] = []
    toks = _tokens(text)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _MULTIPLIERS and i + 1 < len(toks):
            nxt = toks[i + 1]
            ch = str(_UNITS[nxt]) if nxt in _UNITS and _UNITS[nxt] < 10 else (nxt if len(nxt) == 1 else "")
            out.append(ch.upper() * _MULTIPLIERS[t])
            i += 2
            continue
        if t in _UNITS and _UNITS[t] < 10:
            out.append(str(_UNITS[t]))
        elif t in _NATO:
            out.append(_NATO[t])
        elif re.fullmatch(r"[a-z0-9]+", t) and (len(t) == 1 or t.isdigit() or re.fullmatch(r"[a-z]*\d[a-z0-9]*", t)):
            out.append(t.upper())
        i += 1
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# -- categorical ------------------------------------------------------------


def parse_yes_no(text: str) -> bool | None:
    toks = set(_tokens(text))
    yes, no = bool(toks & _YES), bool(toks & _NO)
    if yes == no:
        return None  # neither, or contradictory ("yes... no")
    return yes


def parse_gender(text: str) -> str | None:
    toks = set(_tokens(text))
    if toks & {"female", "woman", "girl", "lady", "mahila"}:
        return "female"
    if toks & {"male", "man", "boy", "gentleman", "purush"}:
        return "male"
    if toks & {"transgender", "other", "non", "nonbinary"}:
        return "other"
    return None


# -- fuzzy text -----------------------------------------------------------------


_TITLES = {"mr", "mrs", "ms", "miss", "shri", "smt", "sri", "dr", "kumari", "my", "name", "is", "i", "am", "this"}


def name_similarity(spoken: str, expected: str) -> float:
    """Order-insensitive fuzzy similarity in [0, 1] between two personal names."""

    def norm(s: str) -> str:
        return " ".join(sorted(t for t in _tokens(s) if t not in _TITLES))

    a, b = norm(spoken), norm(expected)
    ratio = SequenceMatcher(None, a, b).ratio() if a and b else 0.0
    # OCR (or a short-form ID) can give a partial name, e.g. 'Kumar Sharma' for 'Rahul Kumar Sharma'.
    # If every token on the ID was said, that is a match; require >= 2 tokens so a lone surname
    # on the ID can't be satisfied by any name containing it.
    want, said = set(b.split()), set(a.split())
    if len(want) >= 2 and want <= said:
        return max(ratio, 1.0)
    return ratio


def address_overlap(spoken: str, expected: str) -> float:
    """Fraction of the ID address's significant words present in the spoken answer."""
    want = {t for t in _tokens(expected) if len(t) >= 3 or t.isdigit()}
    said = set(_tokens(spoken))
    # also let spoken digit words count towards a PIN code
    said |= set(re.findall(r"\d+", spoken_id(spoken)))
    return len(want & said) / len(want) if want else 0.0

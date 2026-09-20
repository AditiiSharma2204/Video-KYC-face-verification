from datetime import date

import pytest

from videokyc.voice import parsing as p


@pytest.mark.parametrize(
    "text, expected",
    [
        ("thirty two", 32), ("thirty-two years old", 32), ("twenty five", 25), ("forty", 40),
        ("nineteen", 19), ("one hundred and five", 105), ("zero", 0), ("two thousand five", 2005),
        ("I am thirty five years old", 35),
    ],
)
def test_words_to_number(text, expected):
    assert p.words_to_number(text) == expected


@pytest.mark.parametrize("text", ["hello there", "", "my name is Rahul"])
def test_words_to_number_none(text):
    assert p.words_to_number(text) is None


@pytest.mark.parametrize("text, expected", [("32", 32), ("I am 32 years old", 32), ("thirty two", 32), ("no idea", None)])
def test_extract_int(text, expected):
    assert p.extract_int(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "15/08/1990", "15-08-1990", "15.08.1990", "15 08 1990", "15th August 1990", "August 15, 1990",
        "15 august nineteen ninety", "fifteenth of August nineteen ninety", "fifteen august 1990",
        "the 15th of Aug 1990",
    ],
)
def test_parse_date_many_spoken_forms(text):
    assert p.parse_date(text) == date(1990, 8, 15)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("twenty first march two thousand five", date(2005, 3, 21)),
        ("1st January 2000", date(2000, 1, 1)),
        ("thirty first december nineteen eighty five", date(1985, 12, 31)),
        ("5 may 1995", date(1995, 5, 5)),
        ("seventh of june nineteen oh five", date(1905, 6, 7)),
        ("two thousand", None),
    ],
)
def test_parse_date_tricky(text, expected):
    assert p.parse_date(text) == expected


@pytest.mark.parametrize("text", ["31/02/1990", "not a date", "may I say something", "15 august", ""])
def test_parse_date_rejects_invalid_or_incomplete(text):
    assert p.parse_date(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("one two three four five six", "123456"),
        ("A B C D E 1 2 3 4 F", "ABCDE1234F"),
        ("alpha bravo charlie delta echo one two three four foxtrot", "ABCDE1234F"),
        ("double five triple zero", "55000"),
        ("2345 6789 0123", "234567890123"),
        ("abcpe1234f", "ABCPE1234F"),
        ("my number is 1 2 3 4", "1234"),
    ],
)
def test_spoken_id(text, expected):
    assert p.spoken_id(text) == expected


def test_edit_distance():
    assert p.edit_distance("ABCDE1234F", "ABCDE1234F") == 0
    assert p.edit_distance("ABCDE1234F", "ABCDE1284F") == 1
    assert p.edit_distance("ABCDE1234F", "ABCDE123F") == 1
    assert p.edit_distance("", "abc") == 3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("yes", True), ("Yes, I do", True), ("yeah sure", True), ("I consent", True), ("haan", True),
        ("no", False), ("nope", False), ("no I have not", False), ("nahi", False),
        ("maybe", None), ("", None), ("yes no", None), ("I know", None),
    ],
)
def test_yes_no(text, expected):
    assert p.parse_yes_no(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [("male", "male"), ("I am a woman", "female"), ("Female", "female"), ("transgender", "other"), ("cat", None)],
)
def test_gender(text, expected):
    assert p.parse_gender(text) == expected


def test_name_similarity_is_order_and_title_insensitive():
    assert p.name_similarity("Rahul Kumar Sharma", "Rahul Kumar Sharma") == 1.0
    assert p.name_similarity("Sharma Rahul Kumar", "Rahul Kumar Sharma") == 1.0
    assert p.name_similarity("My name is Mr Rahul Kumar Sharma", "Rahul Kumar Sharma") == 1.0
    assert p.name_similarity("Rahul Kumar Sharmaa", "Rahul Kumar Sharma") > 0.9  # small STT slip
    assert p.name_similarity("Amit Verma", "Rahul Kumar Sharma") < 0.5
    assert p.name_similarity("", "Rahul") == 0.0


def test_partial_name_on_id_matches_when_all_its_tokens_are_spoken():
    # OCR dropped the first name: ID says "Kumar Sharma"
    assert p.name_similarity("Rahul Kumar Sharma", "Kumar Sharma") == 1.0
    assert p.name_similarity("Amit Verma", "Kumar Sharma") < 0.6
    # a single surname on the ID is too weak to accept "any name containing it"
    assert p.name_similarity("Amit Sharma", "Sharma") < 0.8


def test_address_overlap():
    expected = "S/O Suresh Kumar, 12 MG Road, Indiranagar, Bengaluru, Karnataka - 560038"
    good = "twelve MG road Indiranagar Bengaluru Karnataka 560038 suresh kumar"
    assert p.address_overlap("12 MG Road Indiranagar Bengaluru Karnataka 560038", expected) >= 0.7
    assert p.address_overlap(good, expected) >= 0.5
    assert p.address_overlap("I live in Mumbai near the station", expected) < 0.2

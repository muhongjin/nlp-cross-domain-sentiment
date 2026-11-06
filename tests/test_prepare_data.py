"""Tests for the text normalisation and duplicate keys used when building the splits."""
from prepare_data import normalise, text_key


def test_normalise_collapses_whitespace_and_case():
    assert normalise("  A  Film\t\nReview ") == "a film review"


def test_text_key_is_stable_across_calls():
    text = "A perfectly ordinary review."
    assert text_key(text) == text_key(text)


def test_text_key_ignores_case_and_spacing():
    assert text_key("Not  bad") == text_key("not bad")


def test_text_key_separates_different_text():
    assert text_key("good film") != text_key("good film, but slow")


def test_text_key_returns_bytes():
    key = text_key("anything")
    assert isinstance(key, bytes)
    assert len(key) == 16

"""Tests for the shingle, signature and similarity helpers in near_duplicates."""
from near_duplicates import JACCARD_THRESHOLD, SIGNATURE_SIZE, jaccard, shingles, signature


def test_shingles_count_for_a_sentence():
    text = "one two three four five six"
    assert shingles(text) == {"one two three four five", "two three four five six"}


def test_shingles_handles_short_text():
    assert shingles("too short") == {"too short"}


def test_jaccard_bounds():
    a = shingles("one two three four five six seven")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles("totally different words in this review")) == 0.0


def test_signature_is_deterministic_and_bounded():
    text = " ".join(f"word{i}" for i in range(200))
    first = signature(shingles(text))
    second = signature(shingles(text))
    assert first == second
    assert len(first) <= SIGNATURE_SIZE


def test_near_duplicate_pair_is_above_threshold():
    base = " ".join(f"word{i}" for i in range(60))
    edited = base.replace("word10", "changed10")
    score = jaccard(shingles(base), shingles(edited))
    assert score >= JACCARD_THRESHOLD

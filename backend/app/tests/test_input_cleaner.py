import pytest

from app.utils.input_cleaner import clean_input


def test_strips_surrounding_whitespace():
    assert clean_input("  Backend Engineer \n", 200) == "Backend Engineer"

def test_empty_required_raises():
    with pytest.raises(ValueError, match="Input cannot be empty"):
        clean_input("", 200)

def test_whitespace_only_required_raises():
    with pytest.raises(ValueError, match="Input cannot be empty"):
        clean_input("   \n\t ", 200)

def test_empty_optional_returns_empty():
    assert clean_input("   ", 200, required=False) == ""

def test_truncates_to_max_length():
    assert clean_input("a" * 300, 200) == "a" * 200

def test_short_input_not_truncated():
    assert clean_input("hello", 200) == "hello"

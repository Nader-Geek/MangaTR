import pytest

from mangatr.languages import (
    ARABIC,
    ENGLISH,
    Direction,
    base_direction,
    get_language,
    letters_in_script,
)


def test_lookup_by_code_and_region():
    assert get_language("ar") is ARABIC
    assert get_language("en-US") is ENGLISH
    with pytest.raises(ValueError):
        get_language("xx")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("مرحبا بالعالم", Direction.RTL),
        ("Hello world", Direction.LTR),
        ("123 ...", Direction.LTR),
        ("مرحبا Hello", Direction.RTL),
        ("Hello مرحبا", Direction.LTR),
        ("", Direction.LTR),
    ],
)
def test_base_direction(text, expected):
    assert base_direction(text) is expected


def test_english_reply_is_not_rejected_for_lacking_arabic():
    assert letters_in_script("What are you doing?!", ENGLISH)
    assert not letters_in_script("What are you doing?!", ARABIC)


def test_arabic_reply_passes_and_untranslated_source_fails():
    assert letters_in_script("ماذا تفعل؟!", ARABIC)
    assert not letters_in_script("何をしているの？！", ARABIC)
    assert not letters_in_script("123 !?", ARABIC)

"""Target and source language specs.

A language is a parameter, never a branch in the code. The prompt names the
language in English, the direction of a text comes from the text itself, and
the reply check is by script, so an English translation is not rejected for
lacking Arabic letters (the old engine's check did exactly that).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class Direction(StrEnum):
    LTR = "ltr"
    RTL = "rtl"


@dataclass(frozen=True, slots=True)
class Language:
    code: str  # BCP 47 primary subtag, e.g. "ar"
    name: str  # the English name the translation prompt uses
    scripts: frozenset[str]  # Unicode name prefixes of its letters, e.g. "ARABIC"
    fonts: tuple[str, ...] = ()  # preferred families; the fallback chain lives in typeset


ARABIC = Language("ar", "Arabic", frozenset({"ARABIC"}))
ENGLISH = Language("en", "English", frozenset({"LATIN"}))

_LANGUAGES = {lang.code: lang for lang in (ARABIC, ENGLISH)}


def get_language(code: str) -> Language:
    primary = code.lower().split("-")[0]
    try:
        return _LANGUAGES[primary]
    except KeyError:
        raise ValueError(f"unsupported language: {code!r}") from None


def base_direction(text: str) -> Direction:
    """The direction of the first strong character, as the Unicode algorithm does.

    Text with no strong character (digits, punctuation) is left to right.
    """
    for char in text:
        bidi = unicodedata.bidirectional(char)
        if bidi == "L":
            return Direction.LTR
        if bidi in ("R", "AL"):
            return Direction.RTL
    return Direction.LTR


def letters_in_script(text: str, language: Language) -> bool:
    """True when the text has at least one letter of the language's script."""
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if any(name.startswith(script) for script in language.scripts):
            return True
    return False

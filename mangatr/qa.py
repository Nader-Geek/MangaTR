"""Quality checks on one translation. None means it passes.

Returns a short code so a repair request can say what was wrong. The checks
are by script, not by language, so an English translation is not judged by
Arabic letters. Sources: Manga-Translator-OCR's repair predicate (leftover
source script, romaji noise, refusal text) and KOMA's detectors (echo,
length ratio).
"""

from __future__ import annotations

import re
import unicodedata

from .languages import Language

_SOURCE_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")
_REFUSAL = re.compile(
    r"\b(i('| a)?m sorry|i cannot|i can't|as an ai|cannot assist|can't assist|"
    r"unable to (translate|help))\b|عذرا|آسف|لا أستطيع",
    re.IGNORECASE,
)
MIN_RATIO = 0.15
MAX_RATIO = 8.0


def _count_source_letters(text: str) -> int:
    n = 0
    for char in text:
        if char.isalpha() and unicodedata.name(char, "").startswith(_SOURCE_SCRIPTS):
            n += 1
    return n


def check(source: str, translation: str, language: Language) -> str | None:
    """ "blank", "refusal", "residue", "echo", or "length"; None when fine."""
    text = translation.strip()
    if not text:
        return "blank"
    if _REFUSAL.search(text) and not _REFUSAL.search(source):
        return "refusal"
    if language.code not in ("ja", "zh", "ko") and _count_source_letters(text):
        return "residue"
    if text == source.strip() and any(c.isalpha() for c in text):
        return "echo"
    ratio = len(text) / max(1, len(source.strip()))
    # a short Japanese source expands a lot into Arabic or English, so only extremes count
    if ratio > MAX_RATIO * 3 or (len(source) > 12 and ratio < MIN_RATIO):
        return "length"
    return None

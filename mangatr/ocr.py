"""Stage: ocr. Phase 1.

Reads:  the image inside each block's frame, with other blocks' lettering blanked.
Writes: source_text, source_lang.
Model:  a `Recognizer` per source language (Japanese first; see `ocr_models`).
Skips locked blocks, and blocks that already have source_text.

It also decides which detections were not lettering. A box whose reading has no
letters in it (or a free-text box read with very low confidence) is removed from
the page here, before anything is cleaned: on a real chapter a lace ornament
scored 0.565 as free text while real free text scored 0.588, so no detector
threshold can tell them apart, but a reader can.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .pipeline import Context
from .scene import Block, BlockKind, Page, Rect

MIN_FREE_CONFIDENCE = 0.25  # a free-text box read below this is treated as not lettering


@dataclass(frozen=True, slots=True)
class Reading:
    text: str
    confidence: float  # 0..1, the recognizer's own estimate


class Recognizer(Protocol):
    """Reads small crops of lettering. `languages` are the source languages it can
    read (BCP 47 primary subtags); one recognizer may serve several."""

    name: str
    languages: frozenset[str]

    def read(self, crops: Sequence[np.ndarray]) -> list[Reading]: ...


def script_language(text: str) -> str | None:
    """The source language a text is written in, judged by its letters: kana means
    Japanese, Hangul Korean, Han alone Chinese, Latin English. None if no letters."""
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name.startswith(("HIRAGANA", "KATAKANA")):
            counts["ja"] += 1
        elif name.startswith("HANGUL"):
            counts["ko"] += 1
        elif name.startswith("CJK"):
            counts["han"] += 1
        elif name.startswith("LATIN"):
            counts["en"] += 1
    if not counts:
        return None
    if counts["ja"]:
        return "ja"
    if counts["ko"]:
        return "ko"
    if counts["han"]:
        return "zh"
    return "en"


def has_text(text: str) -> bool:
    """True if the reading holds at least one letter or digit of any script."""
    return any(char.isalnum() for char in text)


def blank_others(
    crop: np.ndarray, origin: tuple[int, int], frame: Rect, others: Sequence[Rect]
) -> np.ndarray:
    """A copy of `crop` (the page region whose top-left is `origin`) with every other
    box that does not hold this block's centre filled with the crop's border colour,
    so a neighbour's lettering is not read into this block."""
    out = crop.copy()
    left, top = origin
    h, w = crop.shape[:2]
    band = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    colour = np.median(band, axis=0).astype(np.uint8)
    cx, cy = frame.x + frame.w / 2, frame.y + frame.h / 2
    for other in others:
        if other.x <= cx <= other.right and other.y <= cy <= other.bottom:
            continue  # this block sits inside the other one (a box in a box): keep it
        x0, y0 = int(max(0, other.x - left)), int(max(0, other.y - top))
        x1, y1 = int(min(w, other.right - left)), int(min(h, other.bottom - top))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = colour
    return out


class OcrStage:
    name = "ocr"

    def __init__(
        self,
        recognizers: Sequence[Recognizer],
        load_image: Callable[[Page], np.ndarray] | None = None,
        *,
        source_lang: str | None = None,
        drop_non_text: bool = True,
        pad: int = 4,
        min_free_confidence: float = MIN_FREE_CONFIDENCE,
    ) -> None:
        if not recognizers:
            raise ValueError("at least one recognizer is needed")
        self.recognizers = list(recognizers)
        self.load_image = load_image
        self.source_lang = source_lang
        self.drop_non_text = drop_non_text
        self.pad = pad
        self.min_free_confidence = min_free_confidence
        self.dropped: list[str] = []  # ids of blocks removed as not lettering, for the audit

    def _recognizer(self, lang: str | None) -> Recognizer | None:
        for recognizer in self.recognizers:
            if lang is None or lang in recognizer.languages:
                return recognizer
        return None

    def _read(self, recognizer: Recognizer, crops: list[np.ndarray]) -> list[Reading]:
        readings = recognizer.read(crops)
        if len(readings) != len(crops):
            raise RuntimeError(
                f"{recognizer.name} returned {len(readings)} readings for {len(crops)} crops"
            )
        return readings

    def run(self, page: Page, ctx: Context) -> None:
        if self.load_image is None:
            raise RuntimeError("OcrStage.run needs a load_image; use read(page, image) instead")
        self.read(page, self.load_image(page))

    def read(self, page: Page, image: np.ndarray) -> None:
        """Read the page's blocks from `image` (RGB) and drop the ones that are not text."""
        todo = [b for b in page.blocks if not b.locked and b.source_text is None]
        if not todo:
            return
        height, width = image.shape[:2]
        rects = {b.id: b.frame for b in page.blocks}
        crops: list[np.ndarray] = []
        for block in todo:
            f = block.frame
            x0, y0 = max(0, int(f.x) - self.pad), max(0, int(f.y) - self.pad)
            x1 = min(width, int(np.ceil(f.right)) + self.pad)
            y1 = min(height, int(np.ceil(f.bottom)) + self.pad)
            crop = image[y0:y1, x0:x1]
            if crop.size:
                others = [r for key, r in rects.items() if key != block.id]
                crop = blank_others(crop, (x0, y0), f, others)
            else:
                crop = np.zeros((8, 8, 3), np.uint8)  # a box off the page reads as nothing
            crops.append(crop)

        language = self.source_lang
        recognizer = self._recognizer(language)
        if recognizer is None:
            raise RuntimeError(f"no recognizer for source language {language!r}")
        readings = self._read(recognizer, crops)
        if language is None:
            seen = (script_language(r.text) for r in readings)
            guess = Counter(g for g in seen if g).most_common(1)
            if guess and guess[0][0] not in recognizer.languages:
                better = self._recognizer(guess[0][0])
                if better is not None:
                    recognizer, language = better, guess[0][0]
                    readings = self._read(recognizer, crops)
        for block, reading in zip(todo, readings, strict=True):
            text = reading.text.strip()
            block.source_text = text
            block.source_lang = script_language(text) or language
        self._finish(page, todo, readings)

    def _finish(self, page: Page, todo: list[Block], readings: list[Reading]) -> None:
        if not self.drop_non_text:
            return
        gone = {
            b.id
            for b, r in zip(todo, readings, strict=True)
            if not has_text(r.text)
            or (b.kind == BlockKind.FREE_TEXT and r.confidence < self.min_free_confidence)
        }
        if gone:
            page.blocks = [b for b in page.blocks if b.id not in gone]
            self.dropped.extend(sorted(gone))

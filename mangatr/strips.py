"""Where to cut a long strip: at rows with nothing in them.

Adapted from KOMA's strip splitter (klaus-2/koma-studio,
services/splitter_analysis.py, MIT): it projects the page onto one axis and cuts
where the rows are bright. Improved in three ways: a row counts as empty when it
is *uniform* (little spread between its darkest and lightest pixel), so the dark
or coloured gaps of a webtoon qualify and not only white ones; a cut goes in the
middle of the gap, not on its first row; and a cut is chosen near the wanted tile
height instead of the first qualifying row after the minimum (which gave a tile
of whatever length came out).

Cutting in an empty gap means a lettering is never split between two detector
tiles, and the gaps are also the natural panel boundaries of a strip.
"""

from __future__ import annotations

import cv2
import numpy as np

SPREAD = 24  # max - min across a row, in levels, for the row to count as empty
MIN_GAP = 8  # px of consecutive empty rows that make a gap
BLUR_ROWS = 5  # smooth between rows, as KOMA does, so a single stray row does not break a gap


def empty_rows(gray: np.ndarray, spread: int = SPREAD) -> np.ndarray:
    """Boolean per row: nothing is drawn across it."""
    blurred = cv2.GaussianBlur(gray, (1, BLUR_ROWS), 0) if gray.shape[0] >= BLUR_ROWS else gray
    high = cv2.reduce(blurred, 1, cv2.REDUCE_MAX).ravel().astype(np.int16)
    low = cv2.reduce(blurred, 1, cv2.REDUCE_MIN).ravel().astype(np.int16)
    return (high - low) <= spread


def gap_centres(empty: np.ndarray, min_gap: int = MIN_GAP) -> list[int]:
    """Middle row of every run of at least `min_gap` empty rows."""
    centres: list[int] = []
    start: int | None = None
    for row, flag in enumerate(np.append(empty, False)):
        if flag and start is None:
            start = row
        elif not flag and start is not None:
            if row - start >= min_gap:
                centres.append((start + row - 1) // 2)
            start = None
    return centres


def choose_cuts(total: int, centres: list[int], target: int, window: int) -> list[int]:
    """Cut rows about `target` apart: each at the gap nearest the ideal position
    within +-`window`; where there is none, at the ideal position itself."""
    cuts: list[int] = []
    position = 0
    while total - position > target * 1.25:
        ideal = position + target
        near = [c for c in centres if abs(c - ideal) <= window and c > position + target // 2]
        cut = min(near, key=lambda c: abs(c - ideal)) if near else ideal
        cuts.append(cut)
        position = cut
    return cuts


def tiles_from_cuts(total: int, cuts: list[int], overlap: int) -> list[tuple[int, int, int, int]]:
    """(core_start, core_end, crop_start, crop_end) per tile, in the shape
    `detect.merge_tiles` reads: the core is between two cuts, the crop reaches
    `overlap` beyond each of its cut sides."""
    bounds = [0, *cuts, total]
    tiles = []
    for index in range(len(bounds) - 1):
        start, end = bounds[index], bounds[index + 1]
        tiles.append((start, end, max(0, start - overlap), min(total, end + overlap)))
    return tiles

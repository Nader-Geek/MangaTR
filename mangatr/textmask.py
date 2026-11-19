"""An initial text mask for one detected box, from the pixels alone.

The detector gives a box, not the letters. The mask must cover the whole mark
of the lettering (glyph, outline, glow), not only the glyph core: a mask that
stops at the core leaves a ghost ring after cleaning. The recipe:

1. the background colour is the median of the box's border band;
2. pixels far from it (Otsu on the distance, clamped) are ink;
3. small gaps are closed, then the ink is grown by a radius that follows the
   box size, to reach the halo;
4. if nothing sensible comes out (no ink, or ink everywhere), the whole box is
   used: erasing too much inside a detected box is safer than leaving text.

A segmentation model can replace step 1-3 later; the output contract (a boolean
crop and its origin) stays the same.
"""

from __future__ import annotations

import cv2
import numpy as np

from mangatr.halo import absorb_halo
from mangatr.scene import Rect

MIN_DISTANCE = 24  # levels; below this the "ink" is noise
MAX_DISTANCE = 90
MAX_COVERAGE = 0.75  # more ink than this means the background guess failed


def grow_radius(width: float, height: float) -> int:
    """How far to grow the ink to reach its halo: ~6% of the short side, 2..6 px."""
    return int(min(6, max(2, round(min(width, height) * 0.06))))


def _border_band(gray: np.ndarray, band: int) -> np.ndarray:
    band = max(1, min(band, gray.shape[0] // 2, gray.shape[1] // 2))
    inner = np.zeros(gray.shape, bool)
    inner[band:-band, band:-band] = True
    return gray[~inner]


def text_mask(rgb: np.ndarray, frame: Rect, *, pad: int = 4) -> tuple[int, int, np.ndarray]:
    """(x0, y0, mask) where mask is a boolean crop of the page starting at
    (x0, y0), covering the box plus `pad` pixels (clamped to the page)."""
    height, width = rgb.shape[:2]
    x0, y0 = max(0, int(frame.x) - pad), max(0, int(frame.y) - pad)
    x1, y1 = (
        min(width, int(np.ceil(frame.right)) + pad),
        min(height, int(np.ceil(frame.bottom)) + pad),
    )
    crop = rgb[y0:y1, x0:x1]
    whole = np.ones(crop.shape[:2], bool)
    if crop.size == 0 or min(crop.shape[:2]) < 3:
        return x0, y0, whole

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    background = float(np.median(_border_band(gray, max(2, pad))))
    distance = np.abs(gray.astype(np.float32) - background)
    scaled = np.clip(distance, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    limit = float(np.clip(otsu, MIN_DISTANCE, MAX_DISTANCE))
    ink = (distance >= limit).astype(np.uint8)
    if not ink.any():
        return x0, y0, whole

    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close)
    radius = grow_radius(frame.w, frame.h)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = absorb_halo(crop, cv2.dilate(ink, grow).astype(bool))
    if mask.mean() > MAX_COVERAGE:
        return x0, y0, whole
    return x0, y0, mask

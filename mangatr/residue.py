"""Scrub what a model leaves behind inside a mask on a plain surround.

Taken from the old engine's `_scrub_bright_residuals` / `_scrub_dark_residuals`
(85_cleanup.py, MIT, same author): when the pixels around the mask are one
colour, any masked pixel clearly brighter or darker than that colour is leftover
lettering or a model ghost, and is rebuilt from its surroundings. Improved: the
two scrubs are one pass; the fill is estimated from the ring only (the old one
averaged everything outside the mask in the crop, art included); the thresholds
are named; and it returns how many pixels it changed so a run can be audited.

Not run on textured surrounds: there a model's output legitimately differs from
any single colour (old engine: ring colour spread above 22 levels).
"""

from __future__ import annotations

import cv2
import numpy as np

BRIGHT = 18.0  # levels brighter than the surround that count as a ghost
DARK = 28.0  # levels darker (the eye forgives a pale ghost less than a dark one)
MAX_RING_SPREAD = 22.0  # a more varied surround is texture: leave the model's result
MIN_RING = 40
MIN_BAD = 8
BLUR = 21


def _luminance(pixels: np.ndarray) -> np.ndarray:
    return pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114


def scrub_residue(crop: np.ndarray, mask: np.ndarray, ring: np.ndarray) -> tuple[np.ndarray, int]:
    """(crop with ghost pixels of the mask rebuilt, number of pixels changed).
    `ring` is the boolean surround the colour is read from (see `inpaint.ring_of`)."""
    if ring.sum() < MIN_RING or not mask.any():
        return crop, 0
    ring_pixels = crop[ring].astype(np.float32)
    if float(ring_pixels.std(axis=0).mean()) > MAX_RING_SPREAD:
        return crop, 0
    paper = np.median(ring_pixels, axis=0)
    paper_lum = float(_luminance(paper))
    lum = _luminance(crop.astype(np.float32))
    bad = mask & ((lum > paper_lum + BRIGHT) | (lum < paper_lum - DARK))
    if bad.sum() < MIN_BAD:
        return crop, 0
    bad = (cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & mask

    weight = ring.astype(np.float32)
    denominator = cv2.blur(weight, (BLUR, BLUR))
    estimate = np.empty(crop.shape, np.float32)
    for channel in range(3):
        numerator = cv2.blur(crop[..., channel].astype(np.float32) * weight, (BLUR, BLUR))
        plane = np.full(weight.shape, paper[channel], np.float32)
        np.divide(numerator, denominator, out=plane, where=denominator > 1e-4)
        estimate[..., channel] = plane
    out = crop.copy()
    out[bad] = np.clip(np.rint(estimate[bad]), 0, 255).astype(np.uint8)
    return out, int(bad.sum())

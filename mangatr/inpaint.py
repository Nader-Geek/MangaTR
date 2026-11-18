"""Stage: inpaint. Phase 1.

Reads:  the page image, each block's mask, its balloon and its panel.
Writes: clean, the patch under the block's mask.
How:    a router picks the cheapest tier that rebuilds the block's surroundings,
        per block and not per page; the model is last. Tiers: one colour, a
        smooth background (Telea), then the injected model tiers (OpenCV, AOT,
        LaMa). The order of the model tiers is decided by measuring on the
        target GPU (scripts/), not guessed. Only masked pixels ever change.

The cheap tiers are gated strictly on purpose: a fill that is wrong on a
textured background shows as a grey smear, which is worse than the slower
model. The thresholds (10 and 64) come from Koharu's documentation; the ring
width is our own choice. All three are to be verified on real pages.
See docs/DESIGN.md, "Cleaning: a router by cost".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np

RING_WIDTH = 6  # pixels of unmasked surroundings that are read (our choice)
MIN_RING_PIXELS = 32  # a smaller ring cannot be trusted: go to the model
FLAT_MEAN = 10.0  # mean distance from the fit, in levels (Koharu)
FLAT_MAX = 64.0  # no single pixel this far from the fit (Koharu)


class Tier(Protocol):
    """One way to fill a mask. Returns the filled crop, or None if it cannot."""

    name: str

    def fill(self, crop: np.ndarray, mask: np.ndarray, ring: np.ndarray) -> np.ndarray | None: ...


def ring_of(mask: np.ndarray, width: int = RING_WIDTH, valid: np.ndarray | None = None):
    """The unmasked pixels just past the mask, optionally limited to `valid`
    (a balloon or a panel), as a boolean array."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    ring = grown & ~mask
    if valid is not None:
        ring &= valid
    return ring


def _distance(pixels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per pixel, the largest channel difference from the reference."""
    return np.abs(pixels.astype(np.float32) - reference).max(axis=1)


def _is_smooth(distance: np.ndarray) -> bool:
    return bool(distance.mean() <= FLAT_MEAN and distance.max() < FLAT_MAX)


class FlatTier:
    """The ring is one colour: fill with its median."""

    name = "flat"

    def fill(self, crop, mask, ring):
        if ring.sum() < MIN_RING_PIXELS:
            return None
        pixels = crop[ring]
        median = np.median(pixels, axis=0)
        if not _is_smooth(_distance(pixels, median)):
            return None
        out = crop.copy()
        out[mask] = np.round(median).astype(crop.dtype)
        return out


class SmoothTier:
    """The ring is one plane (a gradient): check the fit, then Telea."""

    name = "smooth"

    def fill(self, crop, mask, ring):
        if ring.sum() < MIN_RING_PIXELS:
            return None
        ys, xs = np.nonzero(ring)
        design = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float32)
        pixels = crop[ring].astype(np.float32)
        coefficients, *_ = np.linalg.lstsq(design, pixels, rcond=None)
        if not _is_smooth(_distance(pixels, design @ coefficients)):
            return None
        return cv2.inpaint(crop, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


CHEAP_TIERS: tuple[Tier, ...] = (FlatTier(), SmoothTier())


class Router:
    """Tries the tiers in order; the last one must always succeed."""

    def __init__(self, tiers: Sequence[Tier]) -> None:
        if not tiers:
            raise ValueError("a router needs at least one tier")
        self.tiers = tuple(tiers)

    def clean(
        self, crop: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[np.ndarray, str]:
        """The crop with its masked pixels rebuilt, and the name of the tier used."""
        if not mask.any():
            return crop.copy(), "none"
        ring = ring_of(mask, valid=valid)
        for tier in self.tiers:
            out = tier.fill(crop, mask, ring)
            if out is not None:
                # Only masked pixels come back: nothing outside the mask changes.
                result = crop.copy()
                result[mask] = out[mask]
                return result, tier.name
        raise RuntimeError("no tier could fill the mask; the last tier must always succeed")

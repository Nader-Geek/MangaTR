"""Find the inside of a speech bubble around some lettering.

Ported in idea from the old engine (`_bubble_domain_local`, `_bubble_interior_mask`
in 85_cleanup.py, MIT, our own code): the bubble is the connected region whose
saturation and brightness match the lettering's surroundings; the letters are
holes in it, so the region is filled, and a bubble is only believed when there is
a distinct wall around it. With the interior known, the cleaner can

* cut the mask to the interior, so the corners of a box that lie outside the wall
  are never touched (the wall and the art behind it stay as they were), and
* read the ring from the interior only, which makes the ring flat, so the cheap
  flat tier whitens the bubble exactly instead of a model leaving faint ghosts.
"""

from __future__ import annotations

import cv2
import numpy as np

TOLERANCE = 55.0  # levels of saturation / brightness that still count as the same fill
WALL_TOLERANCE = 35.0  # how different the wall must be from the fill
MIN_WALL_FRACTION = 0.45  # share of the pixels just outside the interior that must differ
MIN_AREA_FRACTION = 0.10  # the interior must be at least this much of the crop
MAX_AREA_FRACTION = 0.92  # more than this means "the whole crop": no bubble, just paper
ERODE = 2  # keep off the wall


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask.astype(np.uint8) * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    flooded = padded.copy()
    cv2.floodFill(
        flooded, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8), (0, 0), 255
    )
    return (cv2.bitwise_or(padded, cv2.bitwise_not(flooded))[1:-1, 1:-1]) > 0


def bubble_interior(rgb: np.ndarray, seed: np.ndarray, *, erode: int = ERODE) -> np.ndarray | None:
    """Boolean interior of the bubble that contains `seed` (a boolean mask of the
    lettering, same size as `rgb`), or None when no bubble is believed."""
    height, width = seed.shape
    if height < 16 or width < 16 or not seed.any():
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation, value = hsv[..., 1], hsv[..., 2]
    # the fill colour is read just around the lettering, not inside the mask: a mask
    # that follows the ink is mostly ink
    around = (cv2.dilate(seed.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~seed
    if around.sum() < 16:
        return None
    s_med, v_med = float(np.median(saturation[around])), float(np.median(value[around]))
    same = (np.abs(saturation - s_med) <= TOLERANCE) & (np.abs(value - v_med) <= TOLERANCE)
    same = cv2.morphologyEx(same.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(same, connectivity=4)
    near = cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    hits = labels[near & (labels > 0)]
    if hits.size == 0:
        return None
    region = labels == np.bincount(hits).argmax()
    if region.sum() < MIN_AREA_FRACTION * height * width:
        return None
    filled = _fill_holes(region)
    if filled.sum() > MAX_AREA_FRACTION * height * width:
        return None
    outside = (cv2.dilate(filled.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~filled
    if outside.sum() >= 30:
        differs = (np.abs(saturation[outside] - s_med) > WALL_TOLERANCE) | (
            np.abs(value[outside] - v_med) > WALL_TOLERANCE
        )
        if differs.mean() < MIN_WALL_FRACTION:
            return None
    if erode:
        filled = cv2.erode(filled.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode) > 0
    return filled if filled.sum() >= 0.08 * height * width else None

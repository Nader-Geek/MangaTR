"""Grow a text mask over the glow or outline of coloured lettering.

Taken from the old engine's `_absorb_text_halo` (85_cleanup.py, MIT, same
author) and improved in two ways. The old one hard-coded warm hues and a
saturation floor of 55, so only red/orange/yellow lettering got its halo
absorbed and a faint glow (saturation 20 to 55, most of it) was missed. Here the
halo is recognised as *paper mixed with the lettering's own colour*: any hue, any
faintness. Kept from the original: look in a band around the mask, take connected
pieces that touch the mask, but not a piece that mostly lies far from it (that is
art of the same colour, such as red hair, not a halo), then close, grow a little
and add to the mask.

Black or grey lettering has no colour to follow, so it is left to the glow rule
of the mask itself (`textmask.grow_radius`).
"""

from __future__ import annotations

import cv2
import numpy as np

BAND = 17  # px around the mask where a halo is looked for
MIN_INK_SATURATION = 60  # the ink must itself be coloured for this to run
MIX_RANGE = (0.10, 1.2)  # share of ink colour in a halo pixel
LEFT_OVER = 0.30  # colour not explained by the paper/ink mix, as a share of their distance
FAR_SHARE = 0.30  # a piece with more than this share far from the mask is art
MIN_HALO_PIXELS = 40
MIN_PIECE = 12


def _colours(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(paper, ink) colours as float RGB: the paper is the median just around the
    mask; the ink is the mean of the third of the masked pixels farthest from it.
    None when the ink is not a colour (black, grey, white: nothing to follow)."""
    around = (cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~mask
    if around.sum() < 16 or mask.sum() < 20:
        return None
    paper = np.median(rgb[around].astype(np.float32), axis=0)
    inside = rgb[mask].astype(np.float32)
    far = np.linalg.norm(inside - paper, axis=1)
    ink = inside[far >= np.percentile(far, 66)].mean(axis=0)
    if np.linalg.norm(ink - paper) < 60:
        return None
    hsv = cv2.cvtColor(np.uint8([[np.rint(ink)]]), cv2.COLOR_RGB2HSV)[0, 0]
    return (paper, ink) if hsv[1] >= MIN_INK_SATURATION else None


def _mixture(rgb: np.ndarray, paper: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Pixels that look like paper mixed with the ink colour: a glow or a soft
    outline is exactly that, however faint (the old engine's fixed hue and
    saturation test missed faint ones). Share of ink `t` in MIX_RANGE and little
    colour left over once that mix is taken out."""
    direction = ink - paper
    length = float(np.linalg.norm(direction))
    delta = rgb.astype(np.float32) - paper
    t = (delta @ direction) / (length * length)
    left_over = np.linalg.norm(delta - t[..., None] * direction, axis=2)
    return (t >= MIX_RANGE[0]) & (t <= MIX_RANGE[1]) & (left_over <= LEFT_OVER * length + 6.0)


def absorb_halo(rgb: np.ndarray, mask: np.ndarray, *, band: int = BAND) -> np.ndarray:
    """`mask` (boolean, same size as `rgb`) grown over the halo of its lettering's
    colour. Returns `mask` unchanged when the lettering has no colour or no halo."""
    if rgb.shape[:2] != mask.shape or not mask.any():
        return mask
    colours = _colours(rgb, mask)
    if colours is None:
        return mask
    same = _mixture(rgb, *colours)

    size = 2 * band + 1
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
    halo = (same & grown & ~mask).astype(np.uint8)
    if halo.sum() < MIN_HALO_PIXELS:
        return mask

    # connected pieces of that colour anywhere near: which of them are art?
    pieces_count, pieces = cv2.connectedComponents((same & ~mask).astype(np.uint8), connectivity=8)
    far_size = 6 * band + 1
    far = (
        same
        & ~mask
        & ~(cv2.dilate(mask.astype(np.uint8), np.ones((far_size, far_size), np.uint8)) > 0)
    )
    total = np.bincount(pieces[pieces > 0], minlength=pieces_count)
    far_count = np.bincount(pieces[far], minlength=pieces_count)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(halo, connectivity=8)
    keep = np.zeros(mask.shape, np.uint8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < MIN_PIECE:
            continue
        owner = int(np.bincount(pieces[labels == index], minlength=pieces_count).argmax())
        if owner and total[owner] > 0 and far_count[owner] > FAR_SHARE * total[owner]:
            continue  # mostly far from the lettering: art, not a halo
        keep[labels == index] = 1
    if not keep.any():
        return mask
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    keep = cv2.dilate(keep, np.ones((7, 7), np.uint8))
    return mask | (keep > 0)

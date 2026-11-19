"""Clean a whole page: cheap tiers per block, a model per window for the rest.

A block first gets the cheap tiers (flat colour, smooth plane) on a small crop.
Blocks that fail go to `model`, grouped into windows by `plan_windows` so one
model call covers neighbours; the context for the model is the page as cleaned
so far. Only masked pixels change. Nothing here knows which model it is: the
tiers are injected, which is what lets the comparison script try each of them.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mangatr.bubble import bubble_interior
from mangatr.inpaint import CHEAP_TIERS, RING_WIDTH, Tier, ring_of
from mangatr.scene import Block
from mangatr.windows import plan_windows

INSIDE = 0.9  # share of a mask that must lie in its bubble for the bubble to be used (Koharu)

BlockMask = tuple[int, int, np.ndarray]  # x0, y0, boolean crop (from textmask.text_mask)


@dataclass(frozen=True, slots=True)
class Step:
    """What happened: tier used and seconds spent, for one block or one window."""

    block_ids: tuple[str, ...]
    tier: str
    seconds: float


def _paste(canvas: np.ndarray, x0: int, y0: int, mask: np.ndarray) -> None:
    """OR a block mask into a boolean page-sized-or-window canvas, clipped."""
    h, w = canvas.shape
    top, left = max(0, y0), max(0, x0)
    bottom, right = min(h, y0 + mask.shape[0]), min(w, x0 + mask.shape[1])
    if bottom > top and right > left:
        canvas[top:bottom, left:right] |= mask[top - y0 : bottom - y0, left - x0 : right - x0]


def clean_page(
    rgb: np.ndarray,
    blocks: Sequence[Block],
    masks: Mapping[str, BlockMask],
    *,
    cheap: Sequence[Tier] = CHEAP_TIERS,
    model: Tier,
    context: int = 256,
    max_pixels: int = 1_000_000,
) -> tuple[np.ndarray, list[Step]]:
    """A cleaned copy of `rgb` and a log of the steps."""
    out = rgb.copy()
    height, width = rgb.shape[:2]
    steps: list[Step] = []
    pending: list[Block] = []
    used: dict[str, BlockMask] = {}  # the masks actually cleaned (cut to the bubble)

    for block in blocks:
        if block.id not in masks:
            continue
        mx0, my0, bmask = masks[block.id]
        margin = RING_WIDTH + 2
        x0, y0 = max(0, mx0 - margin), max(0, my0 - margin)
        x1 = min(width, mx0 + bmask.shape[1] + margin)
        y1 = min(height, my0 + bmask.shape[0] + margin)
        if block.balloon:  # the whole bubble, so its wall is inside the crop
            xs, ys = [p[0] for p in block.balloon], [p[1] for p in block.balloon]
            bx0, by0 = max(0, int(min(xs))), max(0, int(min(ys)))
            bx1, by1 = min(width, int(max(xs)) + 1), min(height, int(max(ys)) + 1)
            if (max(x1, bx1) - min(x0, bx0)) * (max(y1, by1) - min(y0, by0)) <= max_pixels:
                x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
        mask = np.zeros((y1 - y0, x1 - x0), bool)
        _paste(mask, mx0 - x0, my0 - y0, bmask)
        if not mask.any():
            continue
        crop = out[y0:y1, x0:x1]
        interior = bubble_interior(crop, mask) if block.balloon else None
        if interior is not None:
            if (mask & interior).sum() < INSIDE * mask.sum():
                interior = None  # the lettering crosses the wall: not paper, leave it to the model
            else:
                mask &= interior  # never touch the wall or what is behind it
        if not mask.any():
            continue
        # the ring never reads another lettering's pixels (Koharu's `covered`, KOMA's rule)
        others = np.zeros(mask.shape, bool)
        for other in blocks:
            if other.id != block.id and other.id in masks:
                ox, oy, omask = masks[other.id]
                _paste(others, ox - x0, oy - y0, omask)
        valid = ~others if interior is None else interior & ~others
        used[block.id] = (x0, y0, mask.copy())
        ring = ring_of(mask, valid=valid)
        started = time.perf_counter()
        for tier in cheap:
            filled = tier.fill(crop, mask, ring)
            if filled is not None:
                crop[mask] = filled[mask]
                steps.append(Step((block.id,), tier.name, time.perf_counter() - started))
                break
        else:
            pending.append(block)

    for window in plan_windows(pending, width, height, context=context, max_pixels=max_pixels):
        r = window.rect
        x0, y0 = int(r.x), int(r.y)
        x1, y1 = min(width, int(np.ceil(r.right))), min(height, int(np.ceil(r.bottom)))
        mask = np.zeros((y1 - y0, x1 - x0), bool)
        for block_id in window.block_ids:
            mx0, my0, bmask = used[block_id]
            _paste(mask, mx0 - x0, my0 - y0, bmask)
        crop = out[y0:y1, x0:x1]
        started = time.perf_counter()
        filled = model.fill(crop, mask, ring_of(mask))
        if filled is None:
            raise RuntimeError(f"model tier {model.name} must always fill")
        crop[mask] = filled[mask]
        steps.append(Step(window.block_ids, model.name, time.perf_counter() - started))
    return out, steps

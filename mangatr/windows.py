"""Windows: the crops a model sees.

A window is a block's frame plus some context, clamped to the page. Windows
that overlap are merged while the merge stays under a pixel budget, and a
window never spans two panels (see docs/DESIGN.md, "Cleaning: a router by
cost"). Pure geometry: no numpy, no model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .scene import Block, Rect

DEFAULT_CONTEXT = 256
DEFAULT_MAX_PIXELS = 1_000_000


@dataclass(frozen=True, slots=True)
class Window:
    rect: Rect
    block_ids: tuple[str, ...]
    panel: str | None = None


def _expand(frame: Rect, context: float, page_w: float, page_h: float) -> Rect:
    x0 = max(frame.x - context, 0.0)
    y0 = max(frame.y - context, 0.0)
    x1 = min(frame.right + context, page_w)
    y1 = min(frame.bottom + context, page_h)
    return Rect(x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0))


def _union(a: Rect, b: Rect) -> Rect:
    x0, y0 = min(a.x, b.x), min(a.y, b.y)
    return Rect(x0, y0, max(a.right, b.right) - x0, max(a.bottom, b.bottom) - y0)


def _same_panel(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a == b


def plan_windows(
    blocks: Sequence[Block],
    page_w: float,
    page_h: float,
    *,
    context: float = DEFAULT_CONTEXT,
    max_pixels: float = DEFAULT_MAX_PIXELS,
) -> list[Window]:
    """One window per block, then overlapping ones merged under `max_pixels`.

    A window that is already over the budget stays alone; it is never split.
    The result is sorted top to bottom, then left to right, so a run is
    reproducible.
    """
    ordered = sorted(blocks, key=lambda b: (b.frame.y, b.frame.x, b.id))
    windows = [Window(_expand(b.frame, context, page_w, page_h), (b.id,), b.panel) for b in ordered]
    merged = True
    while merged:
        merged = False
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                a, b = windows[i], windows[j]
                if not _same_panel(a.panel, b.panel):
                    continue
                if a.rect.intersection_area(b.rect) <= 0:
                    continue
                union = _union(a.rect, b.rect)
                if union.area > max_pixels:
                    continue
                windows[i] = Window(union, a.block_ids + b.block_ids, a.panel or b.panel)
                del windows[j]
                merged = True
                break
            if merged:
                break
    return sorted(windows, key=lambda w: (w.rect.y, w.rect.x, w.block_ids))

import cv2
import numpy as np

from mangatr.bubble import bubble_interior
from mangatr.clean_page import clean_page
from mangatr.model_tiers import OpenCvTier
from mangatr.scene import Block, BlockKind, Rect


def page_with_bubble(background=None):
    rng = np.random.default_rng(3)
    page = (
        rng.integers(60, 200, (240, 320, 3), dtype=np.uint8) if background is None else background
    )
    cv2.ellipse(page, (160, 120), (110, 90), 0, 0, 360, (252, 252, 252), -1)
    cv2.ellipse(page, (160, 120), (110, 90), 0, 0, 360, (10, 10, 10), 3)
    for i, word in enumerate(("HELLO", "WORLD")):
        cv2.putText(page, word, (110, 110 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    return page


def text_mask_of(page):
    ink = page.max(axis=2) < 90
    mask = np.zeros(ink.shape, bool)
    mask[90:170, 100:230] = ink[90:170, 100:230]
    return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def test_interior_of_a_bubble_over_a_busy_background():
    page = page_with_bubble()
    inside = bubble_interior(page, text_mask_of(page))
    assert inside is not None
    assert inside[120, 100] and not inside[5, 5]  # centre in, far corner out
    assert inside.sum() > 0.5 * np.pi * 110 * 90 * 0.7


def test_plain_paper_is_not_a_bubble():
    page = np.full((200, 200, 3), 250, np.uint8)
    cv2.putText(page, "HI", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    assert bubble_interior(page, page.max(axis=2) < 90) is None


def test_bubble_is_whitened_exactly_and_the_box_corners_outside_are_left_alone():
    page = page_with_bubble()
    mask = text_mask_of(page)
    # a sloppy box mask: the whole 140 x 90 rectangle, corners fall outside the ellipse
    box = np.zeros_like(mask)
    box[75:165, 90:230] = True
    box |= mask
    block = Block(
        id="b1",
        kind=BlockKind.DIALOGUE,
        frame=Rect(90, 75, 140, 90),
        order=0,
        balloon=((50, 30), (270, 30), (270, 210), (50, 210)),
    )
    out, steps = clean_page(page, [block], {"b1": (0, 0, box)}, model=OpenCvTier())
    assert [s.tier for s in steps] == ["flat"]
    inner = out[100:140, 120:200]
    assert (inner == 252).all() or (np.abs(inner.astype(int) - 252) <= 3).all()
    outside_wall = np.ones(page.shape[:2], bool)
    cv2.ellipse(outside_wall.view(np.uint8), (160, 120), (110, 90), 0, 0, 360, 0, -1)
    assert (out[outside_wall] == page[outside_wall]).all()  # nothing behind the wall changed

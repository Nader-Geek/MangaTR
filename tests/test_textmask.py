import cv2
import numpy as np

from mangatr.scene import Rect
from mangatr.textmask import grow_radius, text_mask


def balloon(text_color=0, bg=235):
    page = np.full((200, 300, 3), bg, np.uint8)
    cv2.putText(page, "AB", (90, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (text_color,) * 3, 4)
    return page


def test_dark_text_on_light_is_covered_with_a_margin():
    page = balloon()
    frame = Rect(80, 60, 110, 70)
    x0, y0, mask = text_mask(page, frame)
    ink = np.abs(page[..., 0].astype(int) - 235) > 100
    ys, xs = np.nonzero(ink)
    assert mask[ys - y0, xs - x0].all()  # every letter pixel is masked
    assert mask.mean() < 0.6  # but not the whole box


def test_light_text_on_dark_works_too():
    page = balloon(text_color=250, bg=30)
    _, _, mask = text_mask(page, Rect(80, 60, 110, 70))
    assert 0.02 < mask.mean() < 0.6


def test_an_empty_box_falls_back_to_the_whole_box():
    page = np.full((100, 100, 3), 200, np.uint8)
    _, _, mask = text_mask(page, Rect(20, 20, 40, 40))
    assert mask.all()


def test_crop_is_clamped_to_the_page():
    page = balloon()
    x0, y0, mask = text_mask(page, Rect(-10, -10, 50, 50))
    assert (x0, y0) == (0, 0) and mask.shape[0] <= 44 + 0 and mask.shape[1] <= 44


def test_grow_radius_follows_the_box_and_is_bounded():
    assert grow_radius(10, 10) == 2
    assert grow_radius(1000, 1000) == 6

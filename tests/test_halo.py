import cv2
import numpy as np

from mangatr.halo import absorb_halo


def lettering(color):
    page = np.full((120, 200, 3), 250, np.uint8)
    glow = np.zeros((120, 200), np.uint8)
    cv2.putText(glow, "AB", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 9)
    glow = cv2.GaussianBlur(glow, (0, 0), 3)
    weight = (glow.astype(np.float32) / 255.0)[..., None]
    page = (page * (1 - weight * 0.6) + np.array(color, np.float32) * weight * 0.6).astype(np.uint8)
    cv2.putText(page, "AB", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
    core = np.zeros((120, 200), np.uint8)
    cv2.putText(core, "AB", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 3)
    return page, core > 0


def test_a_glow_of_the_lettering_colour_is_absorbed_for_any_hue():
    for color in ((230, 90, 20), (30, 90, 230), (150, 30, 200)):  # orange, blue, purple
        page, core = lettering(color)
        grown = absorb_halo(page, core)
        assert grown.sum() > core.sum() * 1.15, color
        assert (grown & core).sum() == core.sum()  # only ever adds


def test_black_lettering_is_left_alone():
    page = np.full((120, 200, 3), 250, np.uint8)
    cv2.putText(page, "AB", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    core = page.max(axis=2) < 100
    assert (absorb_halo(page, core) == core).all()


def test_same_coloured_art_far_from_the_lettering_is_not_a_halo():
    page, core = lettering((230, 90, 20))
    page[0:120, 150:200] = (230, 90, 20)  # a big block of the same colour beside it
    grown = absorb_halo(page, core)
    assert not grown[:, 170:].any()

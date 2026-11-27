import cv2
import numpy as np

from mangatr.scene import Rect
from mangatr.textmask import add_glow, grow_radius, text_mask


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


def test_glow_around_light_lettering_is_masked():
    page = np.full((120, 200, 3), 20, np.uint8)
    cv2.putText(page, "I", (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,) * 3, 6)
    glow = cv2.GaussianBlur(page, (0, 0), 5).astype(np.int32)
    shadowed = np.clip(page.astype(np.int32) * 0 + glow * 1, 0, 255).astype(np.uint8)
    shadowed = np.maximum(shadowed, page)
    frame = Rect(70, 40, 50, 60)
    x0, y0, mask = text_mask(shadowed, frame)
    faint = (shadowed[..., 0].astype(int) - 20) >= 14  # visible shadow, not core
    ys, xs = np.nonzero(faint)
    inside = (xs >= x0) & (xs < x0 + mask.shape[1]) & (ys >= y0) & (ys < y0 + mask.shape[0])
    covered = mask[ys[inside] - y0, xs[inside] - x0].mean()
    assert covered > 0.95


def test_glow_stops_at_reach_and_at_background():
    ink = np.zeros((40, 40), np.uint8)
    ink[20, 20] = 1
    distance = np.full((40, 40), 50.0, np.float32)  # everything is "faint"
    grown = add_glow(ink, distance, 60.0, reach=3)
    assert grown.sum() == 49  # a 7x7 square: three steps, no more
    distance[:] = 0.0  # nothing is faint: ink stays as it was
    assert add_glow(ink, distance, 60.0, reach=3).sum() == 1


def _striped_pink():
    """Pink lettering on beige vertical stripes: the grey level of the ink equals the
    stripes', so a grey threshold sees noise (the page-07 case)."""
    import cv2

    img = np.zeros((90, 260, 3), np.uint8)
    for x in range(260):
        img[:, x] = (236, 216, 200) if (x // 6) % 2 == 0 else (214, 196, 182)
    cv2.putText(img, "HELP", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (205, 90, 120), 6)
    return img


def test_coloured_lettering_on_stripes_gets_a_tight_mask_not_the_whole_box():
    img = _striped_pink()
    x0, y0, mask = text_mask(img, Rect(5, 5, 250, 80))
    assert mask.mean() < 0.5  # not the rectangle fallback
    ink = np.abs(img.astype(int) - (205, 90, 120)).sum(axis=2) < 40
    covered = (mask & ink[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]).sum()
    assert covered >= 0.95 * ink.sum()


def test_marks_above_the_glyphs_are_covered_more_than_below():
    img = np.full((80, 80, 3), 255, np.uint8)
    img[30:50, 30:34] = 0
    x0, y0, mask = text_mask(img, Rect(20, 20, 40, 40), pad=4)
    ys = np.flatnonzero(mask.any(axis=1)) + y0
    assert (30 - ys.min()) > (ys.max() - 49)


def test_stroke_variation_tells_lettering_from_a_smear():
    from mangatr.textmask import stroke_variation

    bars = np.zeros((60, 60), np.uint8)
    bars[:, 5:9] = 1
    bars[:, 20:24] = 1
    bars[:, 35:39] = 1
    blob = np.zeros((60, 60), np.uint8)
    blob[5:55, 5:12] = 1
    blob[20:50, 12:50] = 1
    blob[10:15, 30:55] = 1
    assert stroke_variation(bars) < stroke_variation(blob)
    assert stroke_variation(np.zeros((5, 5), np.uint8)) == float("inf")


def _fill_ellipse(ink, centre, axes):
    import cv2

    cv2.ellipse(ink, centre, axes, 0, 0, 360, 1, -1)


def _bars_and_blob(with_blob: bool):
    ink = np.zeros((80, 200), np.uint8)
    for k in range(8):  # eight thin strokes, like lettering
        ink[20:50, 10 + 14 * k : 14 + 14 * k] = 1
    if with_blob:
        _fill_ellipse(ink, (160, 50), (30, 25))
    return ink


def test_a_thick_blob_among_thin_strokes_is_art_and_is_removed():
    from mangatr.textmask import drop_art

    cleaned = drop_art(_bars_and_blob(True))
    assert cleaned[50, 160] == 0  # the blob is gone
    assert cleaned[30, 12] == 1  # the strokes stay


def test_lettering_alone_is_untouched():
    from mangatr.textmask import drop_art

    ink = _bars_and_blob(False)
    assert (drop_art(ink) == ink).all()


def test_with_too_few_marks_nothing_is_dropped():
    from mangatr.textmask import drop_art

    ink = np.zeros((60, 60), np.uint8)
    _fill_ellipse(ink, (30, 30), (25, 25))
    assert (drop_art(ink) == ink).all()


def test_text_beside_a_drawing_gets_a_mask_that_stops_at_the_drawing():
    import cv2

    img = np.full((120, 300, 3), 20, np.uint8)
    cv2.putText(img, "WORDS", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    cv2.ellipse(img, (240, 40), (55, 35), 0, 0, 360, (240, 235, 200), -1)  # an eye on the box edge
    x0, y0, mask = text_mask(img, Rect(5, 25, 290, 60))
    assert not mask[40 - y0, 240 - x0]  # the eye is not masked
    assert mask[45 - y0, 30 - x0 : 120 - x0].any()  # the lettering is

import numpy as np

from mangatr.languages import ARABIC, ENGLISH, Direction
from mangatr.scene import Block, BlockKind, Page, Rect
from mangatr.typeset import Area, Measurer, Typesetter, area_of, best_size, find_font, wrap


def ellipse_points(cx, cy, a, b, n=36):
    t = np.linspace(0, 2 * np.pi, n)
    return tuple((float(cx + a * np.cos(x)), float(cy + b * np.sin(x))) for x in t)


def block(text, balloon=None, frame=None, kind=BlockKind.DIALOGUE):
    frame = frame or Rect(40, 40, 120, 80)
    return Block("b", kind, frame, order=0, balloon=balloon, translation=text)


def test_wrap_breaks_on_words_and_cuts_a_word_that_never_fits():
    measure = lambda text, size, direction: float(len(text) * 10)  # noqa: E731
    assert wrap("aa bb cc", 10, lambda i: 50, measure, Direction.LTR) == ["aa bb", "cc"]
    assert wrap("abcdefghij", 10, lambda i: 40, measure, Direction.LTR) == ["abcd", "efgh", "ij"]
    assert wrap("a", 10, lambda i: 0, measure, Direction.LTR) is None


def test_area_follows_the_balloon_and_is_narrower_at_the_top():
    area = area_of(block("x", ellipse_points(100, 80, 90, 60)), (300, 200))
    assert area is not None
    top = area.band(0, 6)
    middle = area.band(area.height // 2 - 3, area.height // 2 + 3)
    assert middle is not None
    assert top is None or (top[1] - top[0]) < (middle[1] - middle[0])


def test_free_text_area_is_the_frame_a_little_wider():
    area = area_of(block("x", None), (300, 200))
    assert area.x0 < 40 and area.height > 80


def test_text_that_fits_gets_a_big_size_and_long_text_a_smaller_one():
    measurer = Measurer(find_font(ENGLISH), ENGLISH)
    area = area_of(block("x", ellipse_points(100, 80, 90, 60)), (300, 200))
    short = best_size("Hi", area, measurer, Direction.LTR)
    long = best_size("word " * 40, area, measurer, Direction.LTR)
    assert short.fits and short.size > long.size


def test_render_writes_styles_and_only_touches_translated_blocks():
    img = np.full((200, 300, 3), 255, np.uint8)
    translated = block("مرحبا بالعالم", ellipse_points(100, 80, 90, 60))
    plain = Block("p", BlockKind.DIALOGUE, Rect(200, 150, 50, 30), order=1)
    locked = Block(
        "l", BlockKind.DIALOGUE, Rect(200, 100, 50, 30), order=2, translation="x", locked=True
    )
    page = Page("p", "p.png", 300, 200, [translated, plain, locked])
    out = Typesetter(ARABIC).render(img, page)
    assert (out != img).any()
    assert translated.style is not None and translated.style.font_size >= 9
    assert plain.style is None and locked.style is None
    assert (out[150:180, 200:250] == 255).all()  # nothing drawn where nothing was translated


def test_dark_background_takes_light_text_with_no_outline_in_a_balloon():
    img = np.full((200, 300, 3), 20, np.uint8)
    b = block("Hello", ellipse_points(100, 80, 90, 60))
    Typesetter(ENGLISH).render(img, Page("p", "p.png", 300, 200, [b]))
    assert b.style.fill == "#ffffff" and b.style.stroke is None


def test_text_on_art_gets_an_outline():
    img = np.full((200, 300, 3), 200, np.uint8)
    b = block("Hello", None)
    Typesetter(ENGLISH).render(img, Page("p", "p.png", 300, 200, [b]))
    assert b.style.stroke is not None and b.style.stroke_width >= 1


def test_area_band_handles_missing_rows():
    area = Area(0, 0, np.array([-1, 5, 5]), np.array([-1, 50, 50]))
    assert area.band(0, 2) is None and area.band(1, 3) == (5.0, 50.0)

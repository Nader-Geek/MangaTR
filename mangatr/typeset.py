"""Stage: typeset.

Reads:  translation, balloon, frame, the cleaned page image, the language.
Writes: style, and the final page image.
How:    Pillow with the Raqm layout engine (HarfBuzz shaping, FriBiDi), so
        Arabic joins correctly and mixed Arabic/English lines reorder by the
        Unicode bidi rules. The text goes into the free area of its balloon,
        line by line: each line is as wide as the balloon is at that height
        (Manga-Translator-OCR's per-row bands, with a margin). One font size per
        page (the median of the sizes each block could take), shrunk for a
        block that cannot hold it, down to MIN_SIZE. A word wider than its line
        at that size is cut between characters, never hyphenated (Arabic has no
        hyphen). Text on art, with no balloon, gets an outline.
"""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .languages import Direction, Language, base_direction, get_language
from .pipeline import Context
from .scene import Block, Page, Style

MIN_SIZE = 9
MAX_SIZE = 120
LINE_HEIGHT = 1.2
MARGIN = 0.07  # of the smaller side of the area, kept free inside the balloon
FREE_PAD = 0.12  # free text: how much wider than the source lettering the area is
DARK = 110  # mean luminance under which the area takes light text
FONT_ENV = "MANGATR_FONT"

_FONT_CANDIDATES = {
    "ar": (
        "NotoSansArabic-Bold.ttf",
        "NotoNaskhArabic-Bold.ttf",
        "Cairo-Bold.ttf",
        "Amiri-Bold.ttf",
        "NotoSansArabic-Regular.ttf",
        "DejaVuSans-Bold.ttf",
    ),
    "en": ("ComicNeue-Bold.ttf", "Bangers-Regular.ttf", "DejaVuSans-Bold.ttf"),
}
_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / ".local/share/fonts"),
    "/content/fonts",
)


def find_font(language: Language, extra_dirs: Sequence[str | Path] = ()) -> str:
    """A font file for the language. $MANGATR_FONT wins, then the project's
    `fonts/` folder, then the system folders; DejaVu Sans is the last resort
    (it has Arabic and Latin). Raises when none is found."""
    explicit = os.environ.get(FONT_ENV)
    if explicit and Path(explicit).is_file():
        return explicit
    roots = [Path(d) for d in (*extra_dirs, Path(__file__).resolve().parents[1] / "fonts")]
    roots += [Path(d) for d in _FONT_DIRS]
    for name in _FONT_CANDIDATES.get(language.code, _FONT_CANDIDATES["en"]):
        for root in roots:
            if root.is_dir():
                hit = next(root.rglob(name), None)
                if hit is not None:
                    return str(hit)
    raise FileNotFoundError(
        f"no font found for {language.name}; put one in fonts/ or set ${FONT_ENV}"
    )


@dataclass(frozen=True, slots=True)
class Area:
    """Where text may go: per-row free span (left, right) inside a bounding box."""

    x0: int
    y0: int
    lefts: np.ndarray  # length = height; -1 where the row has no free span
    rights: np.ndarray

    @property
    def height(self) -> int:
        return len(self.lefts)

    def band(self, top: float, bottom: float) -> tuple[float, float] | None:
        """The span free on every row of [top, bottom), relative to y0."""
        a, b = max(0, int(top)), min(self.height, int(np.ceil(bottom)))
        if b <= a:
            return None
        lefts, rights = self.lefts[a:b], self.rights[a:b]
        if (lefts < 0).any():
            return None
        left, right = float(lefts.max()), float(rights.min())
        return (left, right) if right > left else None


def _central_run(row: np.ndarray, center: int) -> tuple[int, int] | None:
    """The run of True nearest the center column, as (left, right + 1)."""
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[xs[0], xs[breaks + 1]]
    ends = np.r_[xs[breaks], xs[-1]] + 1
    for s, e in zip(starts, ends, strict=True):
        if s <= center < e:
            return int(s), int(e)
    nearest = int(np.argmin(np.minimum(abs(starts - center), abs(ends - 1 - center))))
    return int(starts[nearest]), int(ends[nearest])


def area_of(block: Block, size: tuple[int, int]) -> Area | None:
    """The free area of a block, in page pixels. Inside its balloon when it has
    one (minus a margin), else the source frame widened a little."""
    width, height = size
    if block.balloon and len(block.balloon) >= 3:
        pts = np.array(block.balloon, np.float32)
        x0, y0 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        x1, y1 = min(width, int(pts[:, 0].max()) + 1), min(height, int(pts[:, 1].max()) + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        shifted = (pts - np.array([x0, y0], np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 1)
        margin = max(3, int(MARGIN * min(mask.shape)))
        mask = cv2.erode(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
        center = (x1 - x0) // 2
        lefts = np.full(mask.shape[0], -1)
        rights = np.full(mask.shape[0], -1)
        for y in range(mask.shape[0]):
            run = _central_run(mask[y] > 0, center)
            if run:
                lefts[y], rights[y] = run
        if (lefts >= 0).sum() < 8:
            return None
        return Area(x0, y0, lefts, rights)
    f = block.frame
    pad_x, pad_y = f.w * FREE_PAD, f.h * FREE_PAD
    x0, y0 = max(0, int(f.x - pad_x)), max(0, int(f.y - pad_y))
    x1, y1 = min(width, int(f.right + pad_x) + 1), min(height, int(f.bottom + pad_y) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return Area(x0, y0, np.zeros(y1 - y0, int), np.full(y1 - y0, x1 - x0, int))


@dataclass(frozen=True, slots=True)
class Layout:
    lines: tuple[str, ...]
    size: int
    line_height: float
    top: float  # of the first line, relative to the area
    fits: bool


class Measurer:
    """Text widths from one font file, cached per size."""

    def __init__(self, font_path: str, language: Language) -> None:
        self.path = font_path
        self.language = language
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(
                self.path, size, layout_engine=ImageFont.Layout.RAQM
            )
        return self._fonts[size]

    def width(self, text: str, size: int, direction: Direction) -> float:
        return self.font(size).getlength(text, direction=direction.value, language=self._lang())

    def _lang(self) -> str | None:
        return self.language.code


def _break_word(word: str, size: int, limit: float, measure, direction: Direction) -> list[str]:
    """A word wider than the line, cut between characters so each piece fits."""
    pieces: list[str] = []
    current = ""
    for char in word:
        if current and measure(current + char, size, direction) > limit:
            pieces.append(current)
            current = char
        else:
            current += char
    if current:
        pieces.append(current)
    return pieces


def wrap(
    text: str, size: int, limits: Callable[[int], float], measure, direction: Direction
) -> list[str] | None:
    """Greedy line wrapping; `limits(i)` is the width of line i. Words that fit nowhere are
    cut between characters. None when a line has no room at all (limit <= 0)."""
    lines: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        for word in paragraph.split():
            limit = limits(len(lines))
            if limit <= 0:
                return None
            trial = f"{current} {word}" if current else word
            if measure(trial, size, direction) <= limit:
                current = trial
                continue
            if current:
                lines.append(current)
                current = ""
                limit = limits(len(lines))
                if limit <= 0:
                    return None
            if measure(word, size, direction) <= limit:
                current = word
            else:
                *head, current = _break_word(word, size, limit, measure, direction)
                lines += head
        if current:
            lines.append(current)
            current = ""
    return lines


def fit(text: str, area: Area, size: int, measurer: Measurer, direction: Direction) -> Layout:
    """Lay the text out at `size`. The first pass guesses the line count from the full
    width, the next two place each line in the band where it will sit."""
    line_h = size * LINE_HEIGHT
    spans = (area.rights - area.lefts)[area.lefts >= 0]
    guess_width = float(np.percentile(spans, 60)) if spans.size else 0.0
    lines = wrap(text, size, lambda i: guess_width, measurer.width, direction) or [text]
    top = max(0.0, (area.height - len(lines) * line_h) / 2)
    for _ in range(3):

        def limit(i: int, top: float = top) -> float:
            band = area.band(top + i * line_h, top + (i + 1) * line_h)
            return (band[1] - band[0]) if band else 0.0

        placed = wrap(text, size, limit, measurer.width, direction)
        if placed is None:
            return Layout(tuple(lines), size, line_h, top, False)
        lines = placed
        new_top = max(0.0, (area.height - len(lines) * line_h) / 2)
        if abs(new_top - top) < 1:
            top = new_top
            break
        top = new_top
    fits = len(lines) * line_h <= area.height and all(
        area.band(top + i * line_h, top + (i + 1) * line_h) is not None for i in range(len(lines))
    )
    if fits:
        for i, line in enumerate(lines):
            band = area.band(top + i * line_h, top + (i + 1) * line_h)
            if band is None or measurer.width(line, size, direction) > band[1] - band[0] + 0.5:
                fits = False
                break
    return Layout(tuple(lines), size, line_h, top, fits)


def best_size(
    text: str, area: Area, measurer: Measurer, direction: Direction, ceiling: int = MAX_SIZE
) -> Layout:
    """The largest size at which the text fits, or the layout at MIN_SIZE."""
    low, high = MIN_SIZE, max(MIN_SIZE, min(ceiling, area.height))
    best = fit(text, area, low, measurer, direction)
    while low < high:
        mid = (low + high + 1) // 2
        trial = fit(text, area, mid, measurer, direction)
        if trial.fits:
            best, low = trial, mid
        else:
            high = mid - 1
    if best.fits or best.size != MIN_SIZE:
        return best
    return fit(text, area, MIN_SIZE, measurer, direction)


def _luminance(rgb: np.ndarray) -> float:
    return float(np.dot(rgb.reshape(-1, 3).mean(axis=0), (0.299, 0.587, 0.114)))


def _hex(value: int) -> str:
    return f"#{value:02x}{value:02x}{value:02x}"


class Typesetter:
    def __init__(self, language: Language | str, font_path: str | None = None) -> None:
        self.language = get_language(language) if isinstance(language, str) else language
        self.measurer = Measurer(font_path or find_font(self.language), self.language)

    def render(self, clean: np.ndarray, page: Page) -> np.ndarray:
        """A copy of `clean` with every translated block drawn; writes each block's style."""
        height, width = clean.shape[:2]
        jobs: list[tuple[Block, Area, str, Direction, Layout]] = []
        for block in page.reading_order():
            if block.locked or not block.translation:
                continue
            area = area_of(block, (width, height))
            if area is None:
                continue
            text = block.translation.strip()
            direction = base_direction(text)
            layout = best_size(text, area, self.measurer, direction)
            jobs.append((block, area, text, direction, layout))
        if not jobs:
            return clean.copy()
        page_size = int(statistics.median(job[4].size for job in jobs))
        image = Image.fromarray(clean)
        draw = ImageDraw.Draw(image)
        for block, area, text, direction, best in jobs:
            layout = (
                best
                if best.size <= page_size
                else fit(text, area, page_size, self.measurer, direction)
            )
            if best.size > page_size and not layout.fits:
                layout = best
            self._draw(draw, clean, block, area, layout, direction)
        return np.asarray(image)

    def _draw(
        self,
        draw: ImageDraw.ImageDraw,
        clean: np.ndarray,
        block: Block,
        area: Area,
        layout: Layout,
        direction: Direction,
    ) -> None:
        x1 = area.x0 + int(np.max(area.rights))
        y1 = area.y0 + area.height
        patch = clean[area.y0 : y1, area.x0 : x1]
        dark = _luminance(patch) < DARK
        fill, outline = ("#ffffff", "#000000") if dark else ("#000000", "#ffffff")
        on_art = not block.balloon
        stroke = max(1, round(layout.size / 9)) if on_art else 0
        font = self.measurer.font(layout.size)
        for i, line in enumerate(layout.lines):
            band_top = layout.top + i * layout.line_height
            band = area.band(band_top, band_top + layout.line_height)
            if band is None:
                left, right = float(np.min(area.lefts[area.lefts >= 0])), float(np.max(area.rights))
            else:
                left, right = band
            cx = area.x0 + (left + right) / 2
            cy = area.y0 + band_top + layout.line_height / 2
            draw.text(
                (cx, cy),
                line,
                font=font,
                fill=fill,
                anchor="mm",
                direction=direction.value,
                language=self.language.code,
                stroke_width=stroke,
                stroke_fill=outline if stroke else None,
            )
        block.style = Style(
            font_size=float(layout.size),
            fill=fill,
            stroke=outline if stroke else None,
            stroke_width=float(stroke),
        )


class TypesetStage:
    """The stage: takes each page's cleaned image from `get_image` and hands the result to
    `put_image`. It does not know where images live."""

    name = "typeset"

    def __init__(
        self,
        typesetter: Typesetter,
        get_image: Callable[[Page], np.ndarray],
        put_image: Callable[[Page, np.ndarray], None],
    ) -> None:
        self.typesetter = typesetter
        self.get_image = get_image
        self.put_image = put_image

    def run(self, page: Page, ctx: Context) -> None:
        self.put_image(page, self.typesetter.render(self.get_image(page), page))

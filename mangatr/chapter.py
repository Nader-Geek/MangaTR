"""Runs a chapter: detect, ocr, clean, translate, typeset, per page, in order.

The only module that knows how the stages are wired for a whole chapter. The
page order is kept: translation is one conversation, so page N is translated
after page N-1, but it runs on its own thread while the next page is detected
and cleaned (the network and the GPU work at the same time). A page that fails
is reported and the chapter goes on; its original image is passed through.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .chapter_translate import ChapterTranslator, PageReport
from .clean_page import clean_page
from .detect import Detector, drop_false_positives, to_blocks
from .inpaint import Tier
from .ocr import OcrStage
from .scene import Page
from .textmask import text_mask
from .typeset import Typesetter

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def list_pages(folder: str | Path) -> list[Path]:
    """Image files of a folder in natural order (2 before 10)."""
    import re

    def key(path: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]

    return sorted(
        (p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES), key=key
    )


@dataclass(slots=True)
class PageResult:
    name: str
    output: Path | None = None
    blocks: int = 0
    translated: int = 0
    error: str | None = None
    seconds: dict[str, float] = field(default_factory=dict)
    report: PageReport | None = None


@dataclass(slots=True)
class ChapterResult:
    pages: list[PageResult]

    @property
    def outputs(self) -> list[Path]:
        return [p.output for p in self.pages if p.output is not None]

    @property
    def failed(self) -> list[PageResult]:
        return [p for p in self.pages if p.error]


class ChapterRunner:
    def __init__(
        self,
        detector: Detector,
        ocr: OcrStage,
        cheap: Sequence[Tier],
        model: Tier,
        translator: ChapterTranslator,
        typesetter: Typesetter,
        *,
        progress: Callable[[str], None] = lambda message: None,
        keep_pages: bool = True,
    ) -> None:
        self.detector, self.ocr = detector, ocr
        self.cheap, self.model = cheap, model
        self.translator, self.typesetter = translator, typesetter
        self.progress = progress
        self.keep_pages = keep_pages

    def _prepare(self, rgb: np.ndarray, name: str, seconds: dict[str, float]):
        """detect + ocr + clean. Returns the page and its cleaned image."""
        started = time.perf_counter()
        found = self.detector.detect(rgb)
        blocks = to_blocks(drop_false_positives(found, rgb.shape[1]))
        page = Page(name, name, rgb.shape[1], rgb.shape[0], blocks)
        seconds["detect"] = time.perf_counter() - started
        started = time.perf_counter()
        self.ocr.dropped = []
        self.ocr.read(page, rgb)
        seconds["ocr"] = time.perf_counter() - started
        started = time.perf_counter()
        masks = {b.id: text_mask(rgb, b.frame) for b in page.blocks}
        cleaned, _ = clean_page(rgb, page.blocks, masks, cheap=self.cheap, model=self.model)
        seconds["clean"] = time.perf_counter() - started
        return page, cleaned

    def run(self, images: Sequence[Path], out_dir: str | Path) -> ChapterResult:
        out = Path(out_dir)
        (out / "pages").mkdir(parents=True, exist_ok=True)
        results = [PageResult(path.name) for path in images]
        pending: list[tuple[int, Page, np.ndarray, Future[PageReport]]] = []
        with ThreadPoolExecutor(max_workers=1) as translating:
            for index, path in enumerate(images):
                result = results[index]
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    result.error = "unreadable image"
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                try:
                    page, cleaned = self._prepare(rgb, path.name, result.seconds)
                except Exception as exc:  # one bad page must not stop the chapter
                    result.error = f"prepare: {exc}"
                    self._pass_through(result, rgb, out, path)
                    continue
                result.blocks = len(page.blocks)
                self.progress(f"{path.name}: {len(page.blocks)} blocks cleaned")
                pending.append((index, page, cleaned, translating.submit(self._translate, page)))
            for index, page, cleaned, future in pending:
                result = results[index]
                try:
                    result.report = future.result()
                    result.translated = result.report.translated
                    started = time.perf_counter()
                    final = self.typesetter.render(cleaned, page)
                    result.seconds["typeset"] = time.perf_counter() - started
                except Exception as exc:
                    result.error = f"translate/typeset: {exc}"
                    final = cleaned
                target = out / f"{Path(page.id).stem}.png"
                cv2.imwrite(str(target), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
                result.output = target
                if self.keep_pages:
                    page.dump(out / "pages" / f"{Path(page.id).stem}.json")
                self.progress(f"{page.id}: translated {result.translated}")
        return ChapterResult(results)

    def _translate(self, page: Page) -> PageReport:
        started = time.perf_counter()
        report = self.translator.translate_page(page)
        self.progress(f"{page.id}: translation {time.perf_counter() - started:.1f}s")
        return report

    @staticmethod
    def _pass_through(result: PageResult, rgb: np.ndarray, out: Path, path: Path) -> None:
        target = out / f"{path.stem}.png"
        cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        result.output = target

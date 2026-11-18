"""Runs the stages in order. The only module that knows the order.

A stage reads some fields of a Page's blocks and writes its own. It never
calls another stage. A stage with nothing to do (OCR on a page of artwork)
does nothing, and that is not a failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .scene import Page


@dataclass(frozen=True, slots=True)
class Context:
    """What a run shares between stages."""

    workdir: Path
    target_lang: str


class Stage(Protocol):
    name: str

    def run(self, page: Page, ctx: Context) -> None: ...


class StageError(RuntimeError):
    """A stage failed on a page. Names both, so a batch can go on with the others."""

    def __init__(self, stage: str, page_id: str) -> None:
        super().__init__(f"stage {stage!r} failed on page {page_id!r}")
        self.stage = stage
        self.page_id = page_id


def run_page(page: Page, stages: Sequence[Stage], ctx: Context) -> None:
    for stage in stages:
        try:
            stage.run(page, ctx)
        except Exception as exc:
            raise StageError(stage.name, page.id) from exc

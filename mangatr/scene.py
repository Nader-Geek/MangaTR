"""Page and Block: the data every stage reads and writes.

This module holds data only. It imports no model, no image library and no
GPU runtime, so it can be tested anywhere. A stage never calls another stage;
it reads some fields of a Block and writes its own (see docs/DESIGN.md,
"Data model"). Masks and cleaned patches are image files kept next to the
page's JSON; a Block stores only their relative paths.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class BlockKind(StrEnum):
    DIALOGUE = "dialogue"
    CAPTION = "caption"
    SFX = "sfx"
    FREE_TEXT = "free_text"
    WATERMARK = "watermark"


class Align(StrEnum):
    CENTER = "center"
    START = "start"
    END = "end"


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned box in page pixels."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def intersection_area(self, other: Rect) -> float:
        w = min(self.right, other.right) - max(self.x, other.x)
        h = min(self.bottom, other.bottom) - max(self.y, other.y)
        return max(w, 0.0) * max(h, 0.0)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, values: list[float]) -> Rect:
        x, y, w, h = values
        return cls(x, y, w, h)


@dataclass(frozen=True, slots=True)
class Style:
    """How the typesetter drew a translation. Written by typeset only."""

    font_size: float | None = None
    fill: str = "#000000"
    stroke: str | None = None
    stroke_width: float = 0.0
    align: Align = Align.CENTER

    def to_dict(self) -> dict[str, Any]:
        return {
            "font_size": self.font_size,
            "fill": self.fill,
            "stroke": self.stroke,
            "stroke_width": self.stroke_width,
            "align": self.align.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Style:
        return cls(
            font_size=data.get("font_size"),
            fill=data.get("fill", "#000000"),
            stroke=data.get("stroke"),
            stroke_width=data.get("stroke_width", 0.0),
            align=Align(data.get("align", Align.CENTER.value)),
        )


Polygon = tuple[tuple[float, float], ...]


@dataclass(slots=True)
class Block:
    """One text of a page, with everything the stages know about it."""

    id: str
    kind: BlockKind
    frame: Rect
    # detect
    order: int | None = None
    mask: str | None = None
    balloon: Polygon | None = None
    panel: str | None = None
    angle: float = 0.0
    vertical: bool = False
    # ocr
    source_text: str | None = None
    source_lang: str | None = None
    # translate
    translation: str | None = None
    target_lang: str | None = None
    # inpaint
    clean: str | None = None
    # typeset
    style: Style | None = None
    # the user: a locked block is left alone by every stage
    locked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "frame": self.frame.to_list(),
            "order": self.order,
            "mask": self.mask,
            "balloon": [list(p) for p in self.balloon] if self.balloon else None,
            "panel": self.panel,
            "angle": self.angle,
            "vertical": self.vertical,
            "source_text": self.source_text,
            "source_lang": self.source_lang,
            "translation": self.translation,
            "target_lang": self.target_lang,
            "clean": self.clean,
            "style": self.style.to_dict() if self.style else None,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        balloon = data.get("balloon")
        style = data.get("style")
        return cls(
            id=data["id"],
            kind=BlockKind(data["kind"]),
            frame=Rect.from_list(data["frame"]),
            order=data.get("order"),
            mask=data.get("mask"),
            balloon=tuple((float(x), float(y)) for x, y in balloon) if balloon else None,
            panel=data.get("panel"),
            angle=data.get("angle", 0.0),
            vertical=data.get("vertical", False),
            source_text=data.get("source_text"),
            source_lang=data.get("source_lang"),
            translation=data.get("translation"),
            target_lang=data.get("target_lang"),
            clean=data.get("clean"),
            style=Style.from_dict(style) if style else None,
            locked=data.get("locked", False),
        )


@dataclass(slots=True)
class Page:
    """A page image and its blocks."""

    id: str
    image: str  # path of the source image, relative to the page file
    width: int
    height: int
    blocks: list[Block] = field(default_factory=list)

    def reading_order(self) -> list[Block]:
        """Blocks in reading order; blocks detect has not ordered come last."""
        return sorted(
            self.blocks,
            key=lambda b: (b.order is None, b.order if b.order is not None else 0),
        )

    def block(self, block_id: str) -> Block:
        for block in self.blocks:
            if block.id == block_id:
                return block
        raise KeyError(block_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "id": self.id,
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Page:
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported page schema version: {version!r}")
        return cls(
            id=data["id"],
            image=data["image"],
            width=data["width"],
            height=data["height"],
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
        )

    def dump(self, path: str | os.PathLike[str]) -> None:
        """Write the page as JSON, atomically, so a crash never leaves half a file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Page:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

"""Stage: detect. Phase 1.

Reads:  the page image.
Writes: Blocks with id, order, kind, frame, balloon, angle, vertical (the mask is
        made by mangatr/textmask.py, the panel later).
Model:  ogkalu/comic-text-and-bubble-detector (RT-DETR v2, ONNX), whose output
        decoding follows the old engine's RTDetrV2ONNXDetector. Koharu's layout
        model is evaluated later (format and licence unverified, see
        docs/DESIGN.md "Risks").
Order:  top to bottom in rows, right to left inside a row for manga (a
        parameter), as Koharu does. Locked blocks are never replaced.

This file keeps the model out of the logic: `decode` and everything after it
take plain arrays, and `Detector` takes an injected session.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from mangatr.scene import Block, BlockKind, Rect

INPUT_SIZE = 640
LABELS = {0: "bubble", 1: "text_bubble", 2: "text_free"}
MIN_SIDE = 12  # px; smaller boxes are noise
MIN_AREA = 400
TILE_HEIGHT = 1680  # long strips are cut into overlapping tiles (old engine's values)
TILE_OVERLAP = 520


@dataclass(frozen=True, slots=True)
class Detection:
    label: str  # "bubble" | "text_bubble" | "text_free"
    score: float
    rect: Rect


def preprocess(rgb: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 page -> (images [1,3,size,size] float32, orig_target_sizes [1,2] int64 as w,h)."""
    height, width = rgb.shape[:2]
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), np.array([[width, height]], dtype=np.int64)


def _squeeze(array: Any) -> np.ndarray:
    array = np.asarray(array)
    return array[0] if array.ndim >= 2 and array.shape[0] == 1 else array


def decode(
    outputs: Sequence[Any], width: int, height: int, *, threshold: float = 0.35
) -> list[Detection]:
    """Turn the three outputs (labels, boxes, scores) into detections in page
    pixels. Boxes are x1,y1,x2,y2; boxes that look normalised (0..1) are scaled."""
    if len(outputs) < 3:
        return []
    labels, boxes, scores = (_squeeze(o) for o in outputs[:3])
    if boxes.ndim == 1:  # labels and boxes swapped
        labels, boxes = boxes, labels
    found: list[Detection] = []
    for label, box, score in zip(labels, boxes, scores, strict=False):
        if float(score) < threshold or len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if max(abs(x1), abs(x2), abs(y1), abs(y2)) <= 1.5:
            x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
        x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
        y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
        w, h = x2 - x1, y2 - y1
        if w < MIN_SIDE or h < MIN_SIDE or w * h < MIN_AREA:
            continue
        name = LABELS.get(int(label))
        if name is None:
            continue
        found.append(Detection(name, float(score), Rect(x1, y1, w, h)))
    return found


def _iou(a: Rect, b: Rect) -> float:
    inter = a.intersection_area(b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(detections: Sequence[Detection], iou_threshold: float = 0.4) -> list[Detection]:
    """Greedy non-maximum suppression within each label (a bubble and the text
    inside it are different things and must both survive)."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: -d.score):
        if all(k.label != det.label or _iou(k.rect, det.rect) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def drop_false_positives(
    detections: Sequence[Detection],
    page_width: int,
    *,
    max_aspect: float = 10.0,
    min_text_free_score: float = 0.5,
    max_width_fraction: float = 0.9,
) -> list[Detection]:
    """Remove text boxes that cannot be lettering: very long thin strips, free
    text spanning the page width, and free text with a weak score. Bubbles are
    never dropped here. Seen on real pages (two webtoon chapters): a 0.39 box
    along a dashed panel border, a 0.60 box across a full-width lace strip and
    a 0.42 box on an ornament of a vest; real free text scored 0.72 to 0.86."""
    kept = []
    for det in detections:
        if det.label != "bubble":
            r = det.rect
            if max(r.w, r.h) / max(1.0, min(r.w, r.h)) > max_aspect:
                continue
            if det.label == "text_free" and (
                det.score < min_text_free_score or r.w > max_width_fraction * page_width
            ):
                continue
        kept.append(det)
    return kept


def plan_tiles(
    height: int, tile_height: int = TILE_HEIGHT, overlap: int = TILE_OVERLAP
) -> list[tuple[int, int, int, int]]:
    """(core_start, core_end, crop_start, crop_end) for a tall page. A page that
    is not much taller than a tile is one tile."""
    if height <= tile_height * 1.25:
        return [(0, height, 0, height)]
    step = max(500, tile_height - overlap)
    tiles = []
    start = 0
    while start < height:
        end = min(start + tile_height, height)
        tiles.append(
            (start, end, max(0, start - (overlap if start else 0)), min(height, end + overlap))
        )
        if end >= height:
            break
        start += step
    return tiles


def merge_tiles(
    per_tile: Sequence[Sequence[Detection]],
    tiles: Sequence[tuple[int, int, int, int]],
    iou_threshold: float = 0.4,
) -> list[Detection]:
    """Shift each tile's detections to page coordinates, keep those whose centre
    lies in the tile's core band (so each text is owned by one tile), then NMS."""
    merged: list[Detection] = []
    for dets, (core_start, core_end, crop_start, _) in zip(per_tile, tiles, strict=True):
        for det in dets:
            top = det.rect.y + crop_start
            centre = top + det.rect.h / 2
            last = core_end == tiles[-1][1]
            if core_start <= centre < core_end or (last and centre >= core_end):
                merged.append(
                    Detection(det.label, det.score, Rect(det.rect.x, top, det.rect.w, det.rect.h))
                )
    return nms(merged, iou_threshold)


def reading_order(rects: Sequence[Rect], *, rtl: bool = True) -> list[int]:
    """Indices of `rects` in reading order: rows top to bottom, a row being the
    boxes whose vertical centres fall within half the row's first box height;
    inside a row right to left for manga (`rtl`), else left to right."""
    remaining = sorted(range(len(rects)), key=lambda i: rects[i].y + rects[i].h / 2)
    order: list[int] = []
    while remaining:
        first = rects[remaining[0]]
        centre = first.y + first.h / 2
        row = [i for i in remaining if abs(rects[i].y + rects[i].h / 2 - centre) <= first.h / 2]
        row.sort(key=lambda i: -(rects[i].x + rects[i].w / 2) if rtl else rects[i].x)
        order.extend(row)
        remaining = [i for i in remaining if i not in row]
    return order


def to_blocks(detections: Sequence[Detection], *, rtl: bool = True) -> list[Block]:
    """Text detections become blocks, numbered in reading order. A text box whose
    centre lies in a bubble gets that bubble (the smallest one) as its balloon."""
    bubbles = [d.rect for d in detections if d.label == "bubble"]
    texts = [d for d in detections if d.label != "bubble"]
    blocks: list[Block] = []
    for index in reading_order([d.rect for d in texts], rtl=rtl):
        det = texts[index]
        cx, cy = det.rect.x + det.rect.w / 2, det.rect.y + det.rect.h / 2
        inside = [b for b in bubbles if b.x <= cx <= b.right and b.y <= cy <= b.bottom]
        balloon = None
        if inside:
            b = min(inside, key=lambda r: r.area)
            balloon = ((b.x, b.y), (b.right, b.y), (b.right, b.bottom), (b.x, b.bottom))
        kind = BlockKind.DIALOGUE if det.label == "text_bubble" else BlockKind.FREE_TEXT
        blocks.append(
            Block(
                id=f"b{len(blocks) + 1}",
                kind=kind,
                frame=det.rect,
                order=len(blocks),
                balloon=balloon,
                vertical=det.rect.h > 1.6 * det.rect.w,
            )
        )
    return blocks


class Detector:
    """The ONNX model behind `decode`. `session` is an InferenceSession or a stand-in."""

    def __init__(self, session: Any, *, threshold: float = 0.35, iou: float = 0.4) -> None:
        names = [item.name for item in session.get_inputs()]
        self._session = session
        self._images = "images" if "images" in names else names[0]
        self._sizes = "orig_target_sizes" if "orig_target_sizes" in names else None
        self.threshold, self.iou = threshold, iou

    def _one(self, rgb: np.ndarray) -> list[Detection]:
        tensor, sizes = preprocess(rgb)
        feeds = {self._images: tensor}
        if self._sizes:
            feeds[self._sizes] = sizes
        outputs = self._session.run(None, feeds)
        height, width = rgb.shape[:2]
        return nms(decode(outputs, width, height, threshold=self.threshold), self.iou)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        tiles = plan_tiles(rgb.shape[0])
        per_tile = [self._one(rgb[start:end]) for _, _, start, end in tiles]
        if len(tiles) == 1:
            return per_tile[0]
        return merge_tiles(per_tile, tiles, self.iou)

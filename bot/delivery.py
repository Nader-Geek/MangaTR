"""Fit a result into Discord's upload limit: recompress to JPEG, then split into parts."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import cv2

DISCORD_LIMIT = 24 * 1024 * 1024  # a margin under the 25 MB of a plain server


def _jpeg(path: Path, quality: int, folder: Path) -> Path:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    target = folder / f"{path.stem}.jpg"
    cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return target


def parts(images: Sequence[Path], folder: Path, limit: int = DISCORD_LIMIT) -> list[Path]:
    """Zip files, each under `limit`. Images go in as they are; if the whole set is too big
    they are recompressed at JPEG quality 90, then 80, 70; a set still too big is split."""
    folder.mkdir(parents=True, exist_ok=True)
    chosen = list(images)
    for quality in (None, 90, 80, 70):
        if quality is not None:
            chosen = [_jpeg(p, quality, folder) for p in images]
        if sum(p.stat().st_size for p in chosen) <= limit:
            return [_write(chosen, folder / "chapter.zip", 1)]
    result: list[Path] = []
    batch: list[Path] = []
    size = 0
    for path in chosen:
        if batch and size + path.stat().st_size > limit:
            result.append(_write(batch, folder / f"chapter_part{len(result) + 1}.zip", 0))
            batch, size = [], 0
        batch.append(path)
        size += path.stat().st_size
    if batch:
        result.append(_write(batch, folder / f"chapter_part{len(result) + 1}.zip", 0))
    return result


def _write(files: Sequence[Path], target: Path, first: int) -> Path:
    start = first or 1
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for number, path in enumerate(files, start=start):
            archive.write(path, f"{number:03d}{path.suffix.lower()}")
    return target

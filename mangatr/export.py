"""Export: ZIP, CBZ and PDF from the finished page images. PSD with separate layers: phase 4."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .strips import choose_cuts, empty_rows, gap_centres

PDF_MAX_HEIGHT = 4000  # px; a taller strip is cut in an empty gap so a reader can show it


def to_zip(images: Sequence[Path], target: str | Path) -> Path:
    """The images in one ZIP, named 001.ext, 002.ext, ... so readers keep the order."""
    target = Path(target)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for number, image in enumerate(images, start=1):
            archive.write(image, f"{number:03d}{Path(image).suffix.lower()}")
    return target


def to_cbz(images: Sequence[Path], target: str | Path) -> Path:
    return to_zip(images, Path(target).with_suffix(".cbz"))


def slices(path: Path, max_height: int = PDF_MAX_HEIGHT) -> list[np.ndarray]:
    """The image, cut into pieces no taller than `max_height`, at empty rows where possible."""
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    height = rgb.shape[0]
    if height <= max_height:
        return [rgb]
    centres = gap_centres(empty_rows(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
    cuts = choose_cuts(height, centres, max_height, window=max_height // 3)
    edges = [0, *sorted(c for c in cuts if 0 < c < height), height]
    return [rgb[a:b] for a, b in zip(edges, edges[1:], strict=False) if b > a]


def to_pdf(images: Sequence[Path], target: str | Path, max_height: int = PDF_MAX_HEIGHT) -> Path:
    target = Path(target)
    pages = [Image.fromarray(p) for image in images for p in slices(Path(image), max_height)]
    if not pages:
        raise ValueError("no images to export")
    pages[0].save(target, "PDF", save_all=True, append_images=pages[1:], resolution=100.0)
    return target


def export(images: Sequence[Path], fmt: str, target: str | Path) -> Path:
    """`fmt` is "zip", "cbz" or "pdf"."""
    if fmt == "zip":
        return to_zip(images, target)
    if fmt == "cbz":
        return to_cbz(images, target)
    if fmt == "pdf":
        return to_pdf(images, target)
    raise ValueError(f"unknown export format: {fmt!r}")

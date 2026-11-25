"""Upload checks and safe extraction.

Archives are untrusted: names are flattened (no path traversal), only images are
taken, the number, the total uncompressed size and every image's dimensions are
capped before anything is decoded (a zip bomb and a 60000-pixel PNG are refused).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
ARCHIVE_SUFFIXES = (".zip", ".cbz")
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_UNCOMPRESSED = 200 * 1024 * 1024
MAX_PIXELS = 60_000_000
MAX_SIDE = 40_000


class UploadError(Exception):
    """The upload is refused; the message is safe to show the user."""


@dataclass(frozen=True, slots=True)
class Upload:
    name: str
    data: bytes


def check_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as exc:
        raise UploadError("an image could not be read") from exc
    if width * height > MAX_PIXELS or max(width, height) > MAX_SIDE:
        raise UploadError(f"an image is too large ({width}x{height})")


def _natural(name: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def extract(uploads: list[Upload], folder: Path, max_pages: int = 100) -> list[Path]:
    """Write the images of the uploads into `folder`, numbered in order. Returns their paths."""
    folder.mkdir(parents=True, exist_ok=True)
    found: list[tuple[str, bytes]] = []
    for upload in uploads:
        if len(upload.data) > MAX_UPLOAD_BYTES:
            raise UploadError(f"{upload.name} is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        lower = upload.name.lower()
        if lower.endswith(ARCHIVE_SUFFIXES):
            try:
                archive = zipfile.ZipFile(io.BytesIO(upload.data))
            except zipfile.BadZipFile as exc:
                raise UploadError(f"{upload.name} is not a valid zip") from exc
            infos = [i for i in archive.infolist() if not i.is_dir()]
            if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
                raise UploadError("the archive is too large once unpacked")
            members = [
                (PurePosixPath(i.filename).name, i)
                for i in infos
                if PurePosixPath(i.filename).suffix.lower() in IMAGE_SUFFIXES
                and not PurePosixPath(i.filename).name.startswith(".")
            ]
            for name, info in sorted(members, key=lambda m: _natural(m[0])):
                found.append((name, archive.read(info)))
        elif lower.endswith(IMAGE_SUFFIXES):
            found.append((PurePosixPath(upload.name).name, upload.data))
        else:
            raise UploadError(f"{upload.name}: send images, a .zip or a .cbz")
        if len(found) > max_pages:
            raise UploadError(f"more than {max_pages} images")
    paths: list[Path] = []
    for number, (name, data) in enumerate(found, start=1):
        check_image(data)
        suffix = PurePosixPath(name).suffix.lower()
        path = folder / f"{number:03d}{suffix}"
        path.write_bytes(data)
        paths.append(path)
    return paths

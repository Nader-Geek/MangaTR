"""Comparison sheets for the cleaning tiers, in the shape of the reference sheets.

    python -m mangatr.compare --images DIR --models-dir DIR --out DIR

For each page: detect, make the text masks, then clean the page with each
variant and write one sheet per page: original | boxes | mask | one column per
variant, each labelled with its time. A `timings.json` next to the sheets has
the numbers. Variants: `cheap+lama` (the router as designed), `lama`, `aot`,
`opencv` (each alone, no cheap tiers). The sheets are for looking at; the
quality judgement is human.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from mangatr.clean_page import clean_page
from mangatr.detect import Detector, to_blocks
from mangatr.inpaint import CHEAP_TIERS, Tier
from mangatr.model_tiers import GPU_FIRST, OnnxInpaintTier, OpenCvTier, load_session
from mangatr.textmask import text_mask

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_SHEET_COLUMN = 700  # px; each column is scaled to this width at most


def label(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def make_sheet(columns: Sequence[tuple[str, np.ndarray]]) -> np.ndarray:
    """Columns side by side, each scaled to the same height, labelled."""
    height = columns[0][1].shape[0]
    scale = min(1.0, MAX_SHEET_COLUMN / max(c.shape[1] for _, c in columns))
    scaled = []
    for name, image in columns:
        size = (max(1, round(image.shape[1] * scale)), max(1, round(height * scale)))
        scaled.append(label(cv2.resize(image, size, interpolation=cv2.INTER_AREA), name))
    return np.hstack(scaled)


def draw_boxes(rgb: np.ndarray, blocks) -> np.ndarray:
    out = rgb.copy()
    for block in blocks:
        f = block.frame
        cv2.rectangle(out, (int(f.x), int(f.y)), (int(f.right), int(f.bottom)), (255, 0, 0), 2)
    return out


def mask_view(rgb: np.ndarray, masks) -> np.ndarray:
    out = (rgb * 0.35).astype(np.uint8)
    for x0, y0, mask in masks.values():
        view = out[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
        view[mask[: view.shape[0], : view.shape[1]]] = (255, 255, 255)
    return out


def variants(
    models_dir: Path, providers: tuple[str, ...]
) -> dict[str, tuple[Sequence[Tier], Tier]]:
    """name -> (cheap tiers, model tier). Missing model files are skipped."""
    found: dict[str, tuple[Sequence[Tier], Tier]] = {"opencv": ((), OpenCvTier())}
    for kind, filename in (("aot", "aot.onnx"), ("lama", "lama-manga-dynamic.onnx")):
        path = models_dir / filename
        if path.exists():
            tier = OnnxInpaintTier(kind, load_session(str(path), providers))
            found[kind] = ((), tier)
            if kind == "lama":
                found["cheap+lama"] = (CHEAP_TIERS, tier)
    return found


def run_page(
    path: Path, detector: Detector, chosen: dict, out_dir: Path, progress: Callable[[str], None]
) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "unreadable"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    started = time.perf_counter()
    blocks = to_blocks(detector.detect(rgb))
    detect_s = time.perf_counter() - started
    masks = {b.id: text_mask(rgb, b.frame) for b in blocks}
    record: dict = {
        "size": [rgb.shape[1], rgb.shape[0]],
        "blocks": len(blocks),
        "detect_s": round(detect_s, 3),
    }
    columns = [
        ("original", rgb),
        (f"boxes ({len(blocks)})", draw_boxes(rgb, blocks)),
        ("mask", mask_view(rgb, masks)),
    ]
    for name, (cheap, model) in chosen.items():
        started = time.perf_counter()
        cleaned, steps = clean_page(rgb, blocks, masks, cheap=cheap, model=model)
        seconds = time.perf_counter() - started
        tiers: dict[str, int] = {}
        for step in steps:
            tiers[step.tier] = tiers.get(step.tier, 0) + len(step.block_ids)
        record[name] = {"seconds": round(seconds, 3), "blocks_by_tier": tiers}
        columns.append((f"{name} {seconds:.2f}s", cleaned))
        progress(f"{path.name} {name} {seconds:.2f}s {tiers}")
    sheet = make_sheet(columns)
    cv2.imwrite(
        str(out_dir / f"{path.stem}_sheet.jpg"),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cleaning comparison sheets.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default="sheets")
    parser.add_argument("--limit", type=int, default=0, help="first N pages; 0 = all")
    parser.add_argument("--cpu", action="store_true", help="do not use CUDA")
    args = parser.parse_args(argv)

    models_dir, out_dir = Path(args.models_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    providers = ("CPUExecutionProvider",) if args.cpu else GPU_FIRST
    detector_path = models_dir / "detector.onnx"
    if not detector_path.exists():
        raise SystemExit(f"{detector_path} is missing; run mangatr.gpu_check first")
    detector = Detector(load_session(str(detector_path), providers))
    chosen = variants(models_dir, providers)
    pages = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        pages = pages[: args.limit]
    timings = {page.name: run_page(page, detector, chosen, out_dir, print) for page in pages}
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"{len(pages)} sheets in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

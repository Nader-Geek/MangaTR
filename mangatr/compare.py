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
from mangatr.detect import Detector, drop_false_positives, to_blocks
from mangatr.inpaint import CHEAP_TIERS, Tier, ring_of
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


def difficulty(rgb: np.ndarray, block, mask_entry) -> float:
    """How busy the background around the lettering is (std of grey in a ring
    around the mask): the higher, the harder the block to rebuild."""
    x0, y0, mask = mask_entry
    full = np.zeros(rgb.shape[:2], bool)
    h, w = rgb.shape[:2]
    full_crop = full[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
    full_crop |= mask[: full_crop.shape[0], : full_crop.shape[1]]
    top, left = max(0, y0 - 16), max(0, x0 - 16)
    bottom, right = min(h, y0 + mask.shape[0] + 16), min(w, x0 + mask.shape[1] + 16)
    window = full[top:bottom, left:right]
    ring = ring_of(window, 12)
    if ring.sum() < 16:
        return 0.0
    gray = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    return float(gray[ring].std())


def crops_sheet(rgb, blocks, masks, cleaned, *, count: int = 8, margin: int = 40, zoom: int = 2):
    """One row per block, the `count` hardest first: original | mask | each variant,
    each crop magnified, so ghosts and smears can be seen."""
    ranked = sorted(blocks, key=lambda b: -difficulty(rgb, b, masks[b.id]))[:count]
    height, width = rgb.shape[:2]
    rows = []
    for block in ranked:
        f = block.frame
        x0, y0 = max(0, int(f.x) - margin), max(0, int(f.y) - margin)
        x1, y1 = min(width, int(f.right) + margin), min(height, int(f.bottom) + margin)
        mx, my, mask = masks[block.id]
        view = (rgb * 0.35).astype(np.uint8)
        view[my : my + mask.shape[0], mx : mx + mask.shape[1]][
            mask[: height - my, : width - mx]
        ] = 255
        tiles = [rgb, view] + list(cleaned.values())
        row = [t[y0:y1, x0:x1] for t in tiles]
        row = [cv2.resize(r, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC) for r in row]
        rows.append(
            label(np.hstack(row), f"{block.id}  " + " | ".join(["original", "mask", *cleaned]))
        )
    if not rows:
        return None
    wide = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 6), (0, wide - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    return np.vstack(rows)


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
    found = detector.detect(rgb)
    kept = drop_false_positives(found, rgb.shape[1])
    blocks = to_blocks(kept)
    detect_s = time.perf_counter() - started
    masks = {b.id: text_mask(rgb, b.frame) for b in blocks}
    record: dict = {
        "size": [rgb.shape[1], rgb.shape[0]],
        "blocks": len(blocks),
        "detect_s": round(detect_s, 3),
    }
    record["detections"] = [
        [d.label, round(d.score, 3)]
        + [round(v) for v in (d.rect.x, d.rect.y, d.rect.w, d.rect.h)]
        + [d in kept]
        for d in found
    ]
    results: dict[str, np.ndarray] = {}
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
        results[name] = cleaned
        progress(f"{path.name} {name} {seconds:.2f}s {tiers}")
    sheet = make_sheet(columns)
    cv2.imwrite(
        str(out_dir / f"{path.stem}_sheet.jpg"),
        cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    crops = crops_sheet(rgb, blocks, masks, results)
    if crops is not None:
        crop_path = out_dir / f"{path.stem}_crops.jpg"
        cv2.imwrite(
            str(crop_path), cv2.cvtColor(crops, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
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

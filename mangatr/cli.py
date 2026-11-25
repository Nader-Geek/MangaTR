"""Command line: translate a folder of pages into a chapter.

    python -m mangatr.cli translate --images pages/ --to ar --provider gemini --out out/

Models are looked for in --models-dir (detector.onnx, aot.onnx, lama-manga-dynamic.onnx;
`python -m mangatr.gpu_check` downloads and checks them). API keys come from the
environment (see mangatr.config). Nothing here is needed to use the library.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mangatr")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("translate", help="translate a folder of pages")
    t.add_argument("--images", required=True, help="folder of page images")
    t.add_argument("--out", default="out")
    t.add_argument("--to", default="ar", help="target language code: ar or en")
    t.add_argument("--from", dest="source", default=None, help="source language, default: guess")
    t.add_argument(
        "--provider",
        default="gemini",
        help="openai, anthropic, gemini, openrouter, ollama; or a comma list as fallbacks",
    )
    t.add_argument("--model", default=None)
    t.add_argument("--glossary", default=None, help="json, tsv or text glossary file")
    t.add_argument("--instructions", default="", help="extra translation instructions")
    t.add_argument("--models-dir", default="models")
    t.add_argument("--font", default=None, help="font file; default: found automatically")
    t.add_argument("--format", choices=["images", "zip", "cbz", "pdf"], default="zip")
    t.add_argument("--limit", type=int, default=0, help="first N pages; 0 = all")
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--clean", default="cheap+aot", help="cleaning variant, e.g. cheap+aot")
    return parser


def run_translate(args: argparse.Namespace) -> int:
    from .chapter import ChapterRunner, list_pages
    from .chapter_translate import ChapterTranslator
    from .compare import variants
    from .config import make_provider
    from .detect import Detector
    from .export import export
    from .glossary import Glossary
    from .languages import get_language
    from .ocr import OcrStage
    from .ocr_models import MangaOcr
    from .runtime import open_session
    from .typeset import Typesetter

    models = Path(args.models_dir)
    detector_path = models / "detector.onnx"
    if not detector_path.exists():
        raise SystemExit(f"{detector_path} is missing; run `python -m mangatr.gpu_check` first")
    pages = list_pages(args.images)
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        raise SystemExit(f"no images in {args.images}")

    language = get_language(args.to)
    chosen, _ = variants(models, cpu=args.cpu)
    if args.clean not in chosen:
        raise SystemExit(f"cleaning variant {args.clean!r} not available; have {sorted(chosen)}")
    cheap, model = chosen[args.clean]
    glossary = Glossary.load(args.glossary) if args.glossary else Glossary()
    translator = ChapterTranslator(
        make_provider(args.provider, args.model), language, glossary, args.instructions
    )
    runner = ChapterRunner(
        Detector(open_session(str(detector_path), prefer_gpu=not args.cpu)),
        OcrStage([MangaOcr.load(device="cpu" if args.cpu else None)], source_lang=args.source),
        cheap,
        model,
        translator,
        Typesetter(language, args.font),
        progress=lambda message: print(message, flush=True),
    )
    out = Path(args.out)
    result = runner.run(pages, out)
    report = {
        "pages": [
            {
                "name": p.name,
                "blocks": p.blocks,
                "translated": p.translated,
                "error": p.error,
                "failed_blocks": p.report.failed if p.report else [],
                "warnings": p.report.warnings if p.report else [],
                "seconds": {k: round(v, 2) for k, v in p.seconds.items()},
            }
            for p in result.pages
        ]
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    if args.format != "images" and result.outputs:
        target = export(result.outputs, args.format, out / f"chapter.{args.format}")
        print(f"exported {target}")
    summary = f"{len(result.outputs)} pages, {len(result.failed)} with problems"
    print(f"{summary}; see {out / 'report.json'}")
    return 1 if result.failed and not result.outputs else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "translate":
        return run_translate(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())

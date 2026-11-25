import json
import zipfile

import cv2
import numpy as np
import pytest

from mangatr import config
from mangatr.chapter import ChapterRunner, list_pages
from mangatr.chapter_translate import ChapterTranslator
from mangatr.detect import Detection
from mangatr.export import export, slices, to_cbz, to_pdf, to_zip
from mangatr.inpaint import CHEAP_TIERS
from mangatr.languages import ARABIC
from mangatr.model_tiers import OpenCvTier
from mangatr.ocr import OcrStage, Reading
from mangatr.scene import Rect
from mangatr.typeset import Typesetter, find_font


class FakeDetector:
    def detect(self, rgb):
        return [Detection("text_bubble", 0.9, Rect(60, 40, 120, 60))]


class FakeReader:
    name = "fake"
    languages = frozenset({"ja"})

    def read(self, crops):
        return [Reading("こんにちは", 0.9) for _ in crops]


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages):
        self.calls += 1
        return json.dumps({"1": "مرحبا بالجميع"})


def make_pages(folder, count=2, height=200):
    folder.mkdir()
    for n in (1, 2, 10)[:count]:
        img = np.full((height, 260, 3), 245, np.uint8)
        cv2.putText(img, "AB", (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
        cv2.imwrite(str(folder / f"{n}.png"), img)
    return folder


def runner(provider=None):
    return ChapterRunner(
        FakeDetector(),
        OcrStage([FakeReader()], source_lang="ja"),
        CHEAP_TIERS,
        OpenCvTier(),
        ChapterTranslator(provider or FakeProvider(), ARABIC),
        Typesetter(ARABIC),
    )


def test_pages_are_listed_in_natural_order(tmp_path):
    make_pages(tmp_path / "in", count=3)
    assert [p.name for p in list_pages(tmp_path / "in")] == ["1.png", "2.png", "10.png"]


def test_chapter_runs_end_to_end_and_changes_the_page(tmp_path):
    pages = list_pages(make_pages(tmp_path / "in"))
    provider = FakeProvider()
    result = runner(provider).run(pages, tmp_path / "out")
    assert not result.failed and len(result.outputs) == 2
    assert result.pages[0].translated == 1 and provider.calls == 2
    before = cv2.imread(str(pages[0]))
    after = cv2.imread(str(result.outputs[0]))
    assert before.shape == after.shape and (before != after).any()
    assert (tmp_path / "out" / "pages" / "1.json").exists()


def test_a_page_that_fails_is_reported_and_the_chapter_goes_on(tmp_path):
    pages = list_pages(make_pages(tmp_path / "in"))
    (tmp_path / "in" / "2.png").write_bytes(b"not an image")
    result = runner().run(pages, tmp_path / "out")
    assert [p.error for p in result.pages] == [None, "unreadable image"]
    assert len(result.outputs) == 1


def test_a_provider_failure_keeps_the_cleaned_page(tmp_path):
    class Broken:
        name = "broken"

        def complete(self, system, messages):
            raise RuntimeError("network down")

    pages = list_pages(make_pages(tmp_path / "in", count=1))
    result = runner(Broken()).run(pages, tmp_path / "out")
    assert "network down" in result.pages[0].error
    assert result.outputs  # the cleaned page is still written


def test_exports(tmp_path):
    images = list_pages(make_pages(tmp_path / "in"))
    z = to_zip(images, tmp_path / "a.zip")
    assert zipfile.ZipFile(z).namelist() == ["001.png", "002.png"]
    assert to_cbz(images, tmp_path / "b.zip").suffix == ".cbz"
    pdf = to_pdf(images, tmp_path / "c.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert export(images, "zip", tmp_path / "d.zip").exists()
    with pytest.raises(ValueError):
        export(images, "rar", tmp_path / "e.rar")


def test_a_tall_strip_is_cut_for_pdf(tmp_path):
    folder = make_pages(tmp_path / "in", count=1, height=2500)
    path = list_pages(folder)[0]
    pieces = slices(path, max_height=1000)
    assert len(pieces) >= 2 and sum(p.shape[0] for p in pieces) == 2500
    assert all(p.shape[0] <= 1500 for p in pieces)


def test_font_is_found():
    assert find_font(ARABIC).endswith((".ttf", ".otf"))


def test_provider_from_environment():
    p = config.make_provider("openai", env={"OPENAI_API_KEY": "a,b"})
    assert p.name.startswith("openai:")
    chain = config.make_provider(
        "gemini,openai", env={"GEMINI_API_KEY": "x", "OPENAI_API_KEY": "y"}
    )
    assert chain.name.startswith("fallback:")
    with pytest.raises(ValueError, match="no API key"):
        config.make_provider("openai", env={})
    with pytest.raises(ValueError, match="unknown provider"):
        config.make_provider("nope", env={})

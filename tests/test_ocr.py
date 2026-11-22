import numpy as np
import pytest

from mangatr.ocr import OcrStage, Reading, blank_others, has_text, script_language
from mangatr.pipeline import Context
from mangatr.scene import Block, BlockKind, Page, Rect


class Fake:
    def __init__(self, name, languages, answers):
        self.name, self.languages, self.answers, self.calls = name, frozenset(languages), answers, 0

    def read(self, crops):
        self.calls += 1
        return list(self.answers)


def page_with(*blocks):
    return Page("p", "p.png", 400, 400, list(blocks))


def block(bid, rect, kind=BlockKind.DIALOGUE, **kw):
    return Block(bid, kind, Rect(*rect), **kw)


CTX = Context(workdir=None, target_lang="ar")  # type: ignore[arg-type]
IMAGE = np.full((400, 400, 3), 255, np.uint8)


def stage(recs, **kw):
    return OcrStage(recs, lambda page: IMAGE, **kw)


def test_script_language():
    assert script_language("食事にしよう") == "ja"
    assert script_language("피식 웃다") == "ko"
    assert script_language("你好") == "zh"
    assert script_language("Hello") == "en"
    assert script_language("…!?") is None


def test_has_text():
    assert has_text("あ") and has_text("3")
    assert not has_text("…!? ・")


def test_reads_blocks_and_skips_locked_and_done():
    page = page_with(
        block("a", (10, 10, 50, 50)),
        block("b", (100, 10, 50, 50), locked=True),
        block("c", (200, 10, 50, 50), source_text="keep"),
    )
    rec = Fake("jp", {"ja"}, [Reading("こんにちは", 0.9)])
    stage([rec]).run(page, CTX)
    assert page.block("a").source_text == "こんにちは"
    assert page.block("a").source_lang == "ja"
    assert page.block("b").source_text is None and page.block("c").source_text == "keep"
    assert rec.calls == 1


def test_a_box_with_no_letters_is_dropped_before_cleaning():
    page = page_with(block("a", (10, 10, 50, 50)), block("b", (100, 10, 50, 50)))
    rec = Fake("jp", {"ja"}, [Reading("あ", 0.9), Reading("…", 0.9)])
    ocr = stage([rec])
    ocr.run(page, CTX)
    assert [b.id for b in page.blocks] == ["a"] and ocr.dropped == ["b"]


def test_free_text_read_with_low_confidence_is_dropped_but_dialogue_is_kept():
    page = page_with(
        block("free", (10, 10, 50, 50), BlockKind.FREE_TEXT),
        block("bubble", (100, 10, 50, 50), BlockKind.DIALOGUE),
    )
    rec = Fake("jp", {"ja"}, [Reading("ふ", 0.1), Reading("ふ", 0.1)])
    stage([rec]).run(page, CTX)
    assert [b.id for b in page.blocks] == ["bubble"]


def test_dropping_can_be_turned_off():
    page = page_with(block("a", (10, 10, 50, 50)))
    stage([Fake("jp", {"ja"}, [Reading("", 0.0)])], drop_non_text=False).run(page, CTX)
    assert len(page.blocks) == 1


def test_the_language_is_guessed_and_the_right_recognizer_rereads():
    page = page_with(block("a", (10, 10, 50, 50)))
    first = Fake("jp", {"ja"}, [Reading("피식", 0.5)])
    korean = Fake("ko", {"ko"}, [Reading("피식 웃다", 0.9)])
    stage([first, korean]).run(page, CTX)
    assert page.block("a").source_text == "피식 웃다" and page.block("a").source_lang == "ko"
    assert korean.calls == 1


def test_a_named_source_language_with_no_recognizer_is_an_error():
    page = page_with(block("a", (10, 10, 50, 50)))
    with pytest.raises(RuntimeError, match="no recognizer"):
        stage([Fake("jp", {"ja"}, [])], source_lang="ko").run(page, CTX)


def test_a_wrong_number_of_readings_is_an_error():
    page = page_with(block("a", (10, 10, 50, 50)), block("b", (100, 10, 50, 50)))
    with pytest.raises(RuntimeError, match="readings"):
        stage([Fake("jp", {"ja"}, [Reading("あ", 1.0)])]).run(page, CTX)


def test_neighbours_are_blanked_unless_they_hold_the_block():
    crop = np.full((60, 120, 3), 200, np.uint8)
    crop[:, 70:] = 0  # the neighbour's dark lettering, at page x 70..120
    frame = Rect(10, 10, 40, 40)
    out = blank_others(crop, (0, 0), frame, [Rect(70, 0, 50, 60)])
    assert (out == 200).all()
    # a big box that holds this block's centre is not blanked
    kept = blank_others(crop, (0, 0), frame, [Rect(0, 0, 120, 60)])
    assert (kept == crop).all()


def test_clean_reading():
    from mangatr.ocr_models import clean_reading

    assert clean_reading("食 事 に し よ う") == "食事にしよう"
    assert clean_reading("そ…") == "そ..."
    assert clean_reading("え・・・・") == "え...."
    assert clean_reading("a  b\n c") == "abc"


def test_read_works_without_a_loader_and_run_needs_one():
    page = page_with(block("a", (10, 10, 50, 50)))
    ocr = OcrStage([Fake("jp", {"ja"}, [Reading("あ", 0.9)])])
    with pytest.raises(RuntimeError, match="load_image"):
        ocr.run(page, CTX)
    ocr.read(page, IMAGE)
    assert page.block("a").source_text == "あ"

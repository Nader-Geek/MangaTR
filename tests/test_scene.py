import json

import pytest

from mangatr.scene import Align, Block, BlockKind, Page, Rect, Style


def make_page() -> Page:
    balloon = ((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0))
    return Page(
        id="p1",
        image="p1.png",
        width=800,
        height=1200,
        blocks=[
            Block(
                id="b2",
                kind=BlockKind.CAPTION,
                frame=Rect(10, 10, 50, 20),
                order=1,
            ),
            Block(
                id="b1",
                kind=BlockKind.DIALOGUE,
                frame=Rect(100, 200, 120, 60),
                order=0,
                balloon=balloon,
                source_text="こんにちは",
                source_lang="ja",
                translation="مرحبا",
                target_lang="ar",
                style=Style(font_size=24.0, align=Align.CENTER),
                locked=True,
            ),
            Block(id="b3", kind=BlockKind.SFX, frame=Rect(0, 0, 5, 5)),
        ],
    )


def test_round_trip_through_dict():
    page = make_page()
    assert Page.from_dict(page.to_dict()) == page


def test_round_trip_through_file(tmp_path):
    page = make_page()
    path = tmp_path / "pages" / "p1.json"
    page.dump(path)
    assert Page.load(path) == page
    assert not list(path.parent.glob("*.tmp"))


def test_file_keeps_non_ascii_text_readable(tmp_path):
    path = tmp_path / "p1.json"
    make_page().dump(path)
    assert "مرحبا" in path.read_text(encoding="utf-8")


def test_reading_order_puts_unordered_blocks_last():
    ids = [b.id for b in make_page().reading_order()]
    assert ids == ["b1", "b2", "b3"]


def test_block_lookup():
    page = make_page()
    assert page.block("b1").translation == "مرحبا"
    with pytest.raises(KeyError):
        page.block("missing")


def test_unknown_schema_version_is_refused():
    data = make_page().to_dict()
    data["version"] = 99
    with pytest.raises(ValueError, match="schema version"):
        Page.from_dict(data)


def test_dump_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    page = make_page()
    path = tmp_path / "p1.json"
    page.dump(path)
    before = path.read_text(encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(json, "dump", boom)
    with pytest.raises(RuntimeError):
        page.dump(path)
    assert path.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.tmp"))


def test_rect_geometry():
    a = Rect(0, 0, 10, 10)
    b = Rect(5, 5, 10, 10)
    assert a.intersection_area(b) == 25
    assert a.intersection_area(Rect(20, 20, 5, 5)) == 0
    assert (a.right, a.bottom, a.area) == (10, 10, 100)

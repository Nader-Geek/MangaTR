import pytest

from mangatr.languages import ARABIC, ENGLISH
from mangatr.scene import Block, BlockKind, Page, Rect
from mangatr.translate import (
    ReplyError,
    apply_reply,
    build_request,
    parse_reply,
    system_prompt,
)


def block(block_id, text, order, kind=BlockKind.DIALOGUE, locked=False):
    return Block(
        id=block_id,
        kind=kind,
        frame=Rect(0, 0, 10, 10),
        order=order,
        source_text=text,
        locked=locked,
    )


def make_page():
    return Page(
        id="p1",
        image="p1.png",
        width=100,
        height=100,
        blocks=[
            block("b3", "third", 2),
            block("b1", "first", 0),
            block("b2", "second", 1, kind=BlockKind.CAPTION),
            block("sfx", "DOOM", 3, kind=BlockKind.SFX),
            block("wm", "site.com", 4, kind=BlockKind.WATERMARK),
            block("empty", "   ", 5),
            block("none", None, 6),
            block("lock", "kept", 7, locked=True),
        ],
    )


def test_request_is_keyed_by_reading_order_and_skips_what_is_not_translated():
    request = build_request(make_page())
    assert request.payload == {"1": "first", "2": "second", "3": "third"}
    assert request.block_ids == ("b1", "b2", "b3")


def test_page_with_nothing_to_translate_has_no_request():
    page = Page(
        id="p", image="p.png", width=1, height=1, blocks=[block("s", "x", 0, BlockKind.SFX)]
    )
    assert build_request(page) is None


def request_of(*texts):
    page = Page(
        id="p",
        image="p.png",
        width=1,
        height=1,
        blocks=[block(f"b{i}", text, i) for i, text in enumerate(texts)],
    )
    return build_request(page)


def test_good_reply_is_parsed():
    request = request_of("a", "b")
    assert parse_reply('{"1": "x", "2": " y "}', request) == {"1": "x", "2": "y"}


def test_reply_in_a_code_fence_is_parsed():
    request = request_of("a")
    raw = '```json\n{"1": "مرحبا"}\n```'
    assert parse_reply(raw, request) == {"1": "مرحبا"}


def test_extra_keys_are_ignored():
    request = request_of("a")
    assert parse_reply('{"1": "x", "9": "junk"}', request) == {"1": "x"}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        ('{"1": "x"}', "missing key '2'"),
        ('{"1": "x", "2": "  "}', "blank"),
        ('{"1": "x", "2": 5}', "blank or non-text"),
        ('{"1": "x", "2": null}', "blank or non-text"),
    ],
)
def test_bad_replies_fail_the_page(raw, message):
    with pytest.raises(ReplyError, match=message):
        parse_reply(raw, request_of("a", "b"))


def test_apply_writes_translation_and_language():
    page = make_page()
    request = build_request(page)
    reply = {"1": "الأول", "2": "الثاني", "3": "الثالث"}
    written = apply_reply(page, request, reply, "ar")
    assert [b.id for b in written] == ["b1", "b2", "b3"]
    assert page.block("b1").translation == "الأول"
    assert page.block("b1").target_lang == "ar"
    assert page.block("sfx").translation is None


def test_text_given_back_unchanged_gets_no_translation():
    page = make_page()
    request = build_request(page)
    reply = {"1": "first", "2": "ثاني", "3": "ثالث"}
    written = apply_reply(page, request, reply, "ar")
    assert page.block("b1").translation is None
    assert page.block("b1").target_lang is None
    assert [b.id for b in written] == ["b2", "b3"]


def test_a_block_locked_after_the_request_is_not_written():
    page = make_page()
    request = build_request(page)
    page.block("b1").locked = True
    apply_reply(page, request, {"1": "x", "2": "y", "3": "z"}, "ar")
    assert page.block("b1").translation is None


def test_system_prompt_names_the_language_and_keeps_instructions():
    arabic = system_prompt(ARABIC, "Use Egyptian colloquial Arabic.")
    assert "into Arabic" in arabic
    assert arabic.endswith("Use Egyptian colloquial Arabic.")
    english = system_prompt(ENGLISH)
    assert "into English" in english
    assert "Arabic" not in english
    assert system_prompt(ENGLISH, "   ") == english

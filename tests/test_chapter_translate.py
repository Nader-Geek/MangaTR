import json

import httpx
import pytest

from mangatr import qa
from mangatr.chapter_translate import ChapterTranslator, TranslateStage
from mangatr.glossary import Glossary
from mangatr.languages import ARABIC, ENGLISH
from mangatr.pipeline import Context
from mangatr.providers import (
    Anthropic,
    FallbackProvider,
    Gemini,
    KeyRing,
    OpenAICompatible,
    ProviderError,
)
from mangatr.scene import Block, BlockKind, Page, Rect


def page(*texts):
    return Page(
        id="p",
        image="p.png",
        width=100,
        height=100,
        blocks=[
            Block(f"b{i}", BlockKind.DIALOGUE, Rect(0, 0, 5, 5), order=i, source_text=t)
            for i, t in enumerate(texts)
        ],
    )


class Script:
    """A provider that answers from a list and records what it was asked."""

    name = "script"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, list(messages)))
        return self.answers.pop(0)


def test_clean_reply_is_applied_and_enters_history():
    p = page("こんにちは", "さようなら")
    script = Script(json.dumps({"1": "مرحبا", "2": "وداعا"}))
    translator = ChapterTranslator(script, ARABIC)
    report = translator.translate_page(p)
    assert [b.translation for b in p.blocks] == ["مرحبا", "وداعا"]
    assert report.translated == 2 and not report.failed
    assert len(translator.history) == 1


def test_bad_json_is_asked_again_and_the_bad_reply_is_not_remembered():
    p = page("はい")
    script = Script("not json", json.dumps({"1": "نعم"}))
    translator = ChapterTranslator(script, ARABIC)
    report = translator.translate_page(p)
    assert p.blocks[0].translation == "نعم"
    assert report.attempts == 2
    assert len(translator.history) == 1
    assert "not usable" in script.calls[1][1][-1]["content"]


def test_residue_line_is_repaired_alone():
    p = page("はい", "いいえ")
    script = Script(
        json.dumps({"1": "نعم", "2": "いいえ"}),
        json.dumps({"2": "لا"}),
    )
    report = ChapterTranslator(script, ARABIC).translate_page(p)
    assert [b.translation for b in p.blocks] == ["نعم", "لا"]
    assert report.repaired == ["b1"]
    assert "rejected" in script.calls[1][1][-1]["content"]


def test_line_that_cannot_be_repaired_is_left_untranslated_and_reported():
    p = page("はい", "いいえ")
    bad = json.dumps({"2": "いいえ"})
    script = Script(json.dumps({"1": "نعم", "2": "いいえ"}), bad, bad, bad)
    report = ChapterTranslator(script, ARABIC).translate_page(p)
    assert p.blocks[1].translation is None
    assert report.failed == ["b1"]
    assert p.blocks[0].translation == "نعم"


def test_unusable_page_after_all_attempts_raises():
    from mangatr.translate import ReplyError

    translator = ChapterTranslator(Script("x", "y", "z"), ARABIC)
    with pytest.raises(ReplyError):
        translator.translate_page(page("はい"))


def test_history_is_trimmed_to_newest_pairs():
    translator = ChapterTranslator(
        Script(*[json.dumps({"1": f"t{i}"}) for i in range(5)]), ENGLISH, history_pairs=2
    )
    for _ in range(5):
        translator.translate_page(page("はい"))
    script = translator.provider
    assert len(script.calls[-1][1]) == 1 + 2 * 2  # two pairs then the new request


def test_glossary_only_sends_entries_present_on_the_page():
    glossary = Glossary({"魔王": "Demon King", "勇者": "Hero"}, ["Excalibur"])
    section = glossary.prompt_section(["魔王が来る"])
    assert "Demon King" in section and "Hero" not in section and "Excalibur" not in section
    assert "Excalibur" in glossary.prompt_section(["Excalibur!"])
    assert "excalibur" not in glossary.prompt_section(["Excalibur2"]).lower()


def test_glossary_formats():
    assert Glossary.parse('{"魔王": "Demon King"}').terms == {"魔王": "Demon King"}
    g = Glossary.parse('{"terms": {"a": "b"}, "never": ["X"]}')
    assert g.terms == {"a": "b"} and g.never == ["X"]
    t = Glossary.parse("# c\n魔王\tDemon King\nHero = بطل\nKeepMe\n")
    assert t.terms == {"魔王": "Demon King", "Hero": "بطل"} and t.never == ["KeepMe"]


def test_stage_collects_reports():
    p = page("はい")
    stage = TranslateStage(ChapterTranslator(Script(json.dumps({"1": "نعم"})), ARABIC))
    stage.run(p, Context(workdir=None, target_lang="ar"))
    assert len(stage.reports) == 1 and stage.reports[0].translated == 1


@pytest.mark.parametrize(
    ("source", "translation", "expected"),
    [
        ("はい", "نعم", None),
        ("はい", "", "blank"),
        ("はい", "はい", "residue"),
        ("はい", "I'm sorry, I cannot help", "refusal"),
        ("OK", "OK", "echo"),
    ],
)
def test_qa(source, translation, expected):
    language = ENGLISH if translation.isascii() else ARABIC
    assert qa.check(source, translation, language) == expected


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_provider_retries_then_parses():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    sleeps = []
    provider = OpenAICompatible(["k1", "k2"], "m", client=make_client(handler), sleep=sleeps.append)
    assert provider.complete("sys", [{"role": "user", "content": "x"}]) == "hi"
    assert len(seen) == 2
    assert seen[0].headers["authorization"] != seen[1].headers["authorization"]  # key rotated


def test_anthropic_and_gemini_shapes():
    def anthropic_handler(request):
        body = json.loads(request.content)
        assert body["system"] == "sys" and request.headers["x-api-key"] == "k"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "a"}]})

    def gemini_handler(request):
        body = json.loads(request.content)
        assert body["contents"][0]["role"] == "user"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "g"}]}}]})

    msgs = [{"role": "user", "content": "x"}]
    assert Anthropic(["k"], "m", client=make_client(anthropic_handler)).complete("sys", msgs) == "a"
    assert Gemini(["k"], "m", client=make_client(gemini_handler)).complete("sys", msgs) == "g"


def test_client_error_is_not_retried_and_fallback_moves_on():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    bad = OpenAICompatible(["k"], "m", client=make_client(handler), sleep=lambda s: None)
    with pytest.raises(ProviderError):
        bad.complete("s", [{"role": "user", "content": "x"}])
    assert len(calls) == 1
    good = Script("ok")
    assert FallbackProvider([bad, good]).complete("s", []) == "ok"


def test_retries_exhausted_raises():
    provider = OpenAICompatible(
        ["k"],
        "m",
        client=make_client(lambda r: httpx.Response(503)),
        retries=2,
        sleep=lambda s: None,
    )
    with pytest.raises(ProviderError, match="exhausted"):
        provider.complete("s", [])


def test_key_ring_rests_a_limited_key():
    now = [0.0]
    ring = KeyRing(["a", "b"], cooldown=10, clock=lambda: now[0])
    key, wait = ring.next()
    ring.rest(key)
    other, wait = ring.next()
    assert other != key and wait == 0
    ring.rest(other)
    _, wait = ring.next()
    assert wait > 0

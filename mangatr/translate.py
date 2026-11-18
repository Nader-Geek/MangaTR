"""Stage: translate. Phases 1 and 2.

Reads:  source_text of the page's blocks in reading order, the glossary.
Writes: translation, target_lang.
How:    one conversation per chapter, one turn per page. A page is a JSON
        object keyed by reading order and the reply must keep the keys; a
        missing key or a blank translation fails the page and never enters the
        history. The system prompt is fixed, in English, and names the target
        language by its English name.

This module holds the parts that need no network: building a page's request,
checking and applying the reply. The conversation and the providers come
later in this phase. Phase 2 adds the glossary, never-translate masking and
the QA detectors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .languages import Language
from .scene import Block, BlockKind, Page

TRANSLATED_KINDS = frozenset({BlockKind.DIALOGUE, BlockKind.CAPTION, BlockKind.FREE_TEXT})

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)


class ReplyError(ValueError):
    """The reply cannot be used. The page fails and the reply never enters the history."""


@dataclass(frozen=True, slots=True)
class PageRequest:
    """What goes to the model for one page.

    `payload` maps "1", "2", ... (reading order) to the source texts, and
    `block_ids[i]` is the block behind key `str(i + 1)`.
    """

    payload: dict[str, str]
    block_ids: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, indent=2)


def system_prompt(language: Language, instructions: str = "") -> str:
    """The fixed system prompt. It names the target by its English name, so the
    same template serves every language. The user's instructions follow as written."""
    prompt = (
        f"You translate manga and comic text into {language.name}.\n"
        "You receive a JSON object. Each key is the reading order of one text on a page, "
        "and each value is the source text.\n"
        "Reply with only a JSON object that has exactly the same keys. Each value is the "
        f"{language.name} translation of that text. Keep the voice and the tone of the speaker. "
        "Never leave a value empty, never add keys, and add no commentary."
    )
    if instructions.strip():
        prompt += "\n\n" + instructions.strip()
    return prompt


def build_request(page: Page) -> PageRequest | None:
    """The page's texts in reading order, or None when there is nothing to translate."""
    payload: dict[str, str] = {}
    ids: list[str] = []
    for block in page.reading_order():
        text = (block.source_text or "").strip()
        if block.locked or block.kind not in TRANSLATED_KINDS or not text:
            continue
        ids.append(block.id)
        payload[str(len(ids))] = text
    if not ids:
        return None
    return PageRequest(payload, tuple(ids))


def parse_reply(raw: str, request: PageRequest) -> dict[str, str]:
    """The reply as {key: translation}, or ReplyError.

    A code fence around the JSON is tolerated. Every key of the request must be
    present with a non-blank string. Extra keys are ignored.
    """
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReplyError(f"reply is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ReplyError("reply is not a JSON object")
    reply: dict[str, str] = {}
    for key in request.payload:
        if key not in data:
            raise ReplyError(f"reply is missing key {key!r}")
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ReplyError(f"reply has a blank or non-text value for key {key!r}")
        reply[key] = value.strip()
    return reply


def apply_reply(
    page: Page, request: PageRequest, reply: dict[str, str], target_lang: str
) -> list[Block]:
    """Write the translations into the page. Returns the blocks it wrote.

    A text given back unchanged (a copyright line, say) gets no translation, so
    it keeps its own lettering. A locked block is never written.
    """
    written: list[Block] = []
    for index, block_id in enumerate(request.block_ids, start=1):
        block = page.block(block_id)
        if block.locked:
            continue
        translation = reply[str(index)]
        if translation == request.payload[str(index)]:
            block.translation = None
            block.target_lang = None
            continue
        block.translation = translation
        block.target_lang = target_lang
        written.append(block)
    return written

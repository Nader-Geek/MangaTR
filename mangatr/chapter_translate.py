"""One conversation per chapter, one turn per page, with repair.

A page that comes back unusable (bad JSON, a missing key) is asked again with
the problem named, up to `attempts` times; the failed reply never enters the
history. A usable reply whose lines fail the quality checks (leftover source
script, refusal, ...) gets a repair request for those lines only. Lines that
still fail are left untranslated and reported, never drawn. History keeps the
newest pages within a character budget, so the prefix stays stable between
turns (prompt caching).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import qa
from .glossary import Glossary
from .languages import Language
from .pipeline import Context
from .providers import Provider
from .scene import Page
from .translate import (
    PageRequest,
    ReplyError,
    apply_reply,
    build_request,
    parse_reply,
    system_prompt,
)

HISTORY_PAIRS = 12
HISTORY_CHARS = 8000
HARD = frozenset({"blank", "refusal", "residue"})


@dataclass(slots=True)
class PageReport:
    page_id: str
    translated: int = 0
    attempts: int = 0
    repaired: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # block ids left untranslated
    warnings: list[str] = field(default_factory=list)


class ChapterTranslator:
    def __init__(
        self,
        provider: Provider,
        language: Language,
        glossary: Glossary | None = None,
        instructions: str = "",
        *,
        attempts: int = 3,
        history_pairs: int = HISTORY_PAIRS,
        history_chars: int = HISTORY_CHARS,
    ) -> None:
        self.provider = provider
        self.language = language
        self.glossary = glossary or Glossary()
        self.system = system_prompt(language, instructions)
        self.attempts = attempts
        self.history_pairs = history_pairs
        self.history_chars = history_chars
        self.history: list[tuple[str, str]] = []

    def _messages(self, user: str, extra: list[dict[str, str]] | None = None):
        pairs = self.history[-self.history_pairs :]
        while len(pairs) > 1 and sum(len(a) + len(b) for a, b in pairs) > self.history_chars:
            pairs = pairs[1:]
        messages: list[dict[str, str]] = []
        for asked, answered in pairs:
            messages += [
                {"role": "user", "content": asked},
                {"role": "assistant", "content": answered},
            ]
        messages.append({"role": "user", "content": user})
        return messages + (extra or [])

    def _user(self, payload: dict[str, str]) -> str:
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        glossary = self.glossary.prompt_section(payload.values())
        return f"{glossary}\n\n{body}" if glossary else body

    def _ask(self, request: PageRequest, report: PageReport) -> dict[str, str]:
        user = self._user(request.payload)
        extra: list[dict[str, str]] = []
        error = "no attempt"
        for _ in range(self.attempts):
            report.attempts += 1
            raw = self.provider.complete(self.system, self._messages(user, extra))
            try:
                reply = parse_reply(raw, request)
            except ReplyError as exc:
                error = str(exc)
                extra = [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That reply was not usable: {error}. "
                        "Reply again with only the JSON object and every key.",
                    },
                ]
                continue
            self.history.append((user, json.dumps(reply, ensure_ascii=False, indent=2)))
            return reply
        raise ReplyError(error)

    def _repair(
        self, request: PageRequest, reply: dict[str, str], bad: dict[str, str], report: PageReport
    ) -> None:
        subset = {key: request.payload[key] for key in bad}
        sub = PageRequest(subset, tuple(request.block_ids[int(k) - 1] for k in subset))
        notes = ", ".join(f"{k}: {why}" for k, why in bad.items())
        user = (
            self._user(subset)
            + f"\n\nTranslate these again. The earlier translation was rejected ({notes}). "
            f"Write only {self.language.name}, no source-language text, no commentary."
        )
        for _ in range(self.attempts):
            report.attempts += 1
            raw = self.provider.complete(self.system, self._messages(user))
            try:
                fixed = parse_reply(raw, sub)
            except ReplyError:
                continue
            for key, text in fixed.items():
                if qa.check(request.payload[key], text, self.language) not in HARD:
                    reply[key] = text
                    report.repaired.append(request.block_ids[int(key) - 1])
                    bad.pop(key, None)
            if not bad:
                return

    def translate_page(self, page: Page) -> PageReport:
        report = PageReport(page.id)
        request = build_request(page)
        if request is None:
            return report
        reply = self._ask(request, report)
        verdicts = {
            key: why
            for key, text in reply.items()
            if (why := qa.check(request.payload[key], text, self.language))
        }
        hard = {key: why for key, why in verdicts.items() if why in HARD}
        if hard:
            self._repair(request, reply, hard, report)
        for key, why in verdicts.items():
            if key not in hard and why != "echo":
                report.warnings.append(f"{request.block_ids[int(key) - 1]}: {why}")
        # lines still failing are left untranslated, never drawn
        for key in hard:
            report.failed.append(request.block_ids[int(key) - 1])
            reply[key] = request.payload[key]  # apply_reply treats an echo as "no translation"
        written = apply_reply(page, request, reply, self.language.code)
        report.translated = len(written)
        return report


class TranslateStage:
    name = "translate"

    def __init__(self, translator: ChapterTranslator) -> None:
        self.translator = translator
        self.reports: list[PageReport] = []

    def run(self, page: Page, ctx: Context) -> None:
        self.reports.append(self.translator.translate_page(page))

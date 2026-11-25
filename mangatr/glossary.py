"""Glossary and never-translate list.

Only the entries that appear in a page's texts go into its request, so a long
glossary costs nothing on a page that does not use it (the BallonsTranslator and
OverTranslate rule). Longest keys match first; a Latin key matches whole words
only. Formats: JSON ({"never": [...], "terms": {src: dst}} or a flat {src: dst}),
TSV ("src<TAB>dst", a line with no tab is a never-translate term), or plain text
with "src = dst" lines.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

MAX_TERMS = 20
MAX_CHARS = 600


def _found(key: str, text: str) -> bool:
    if key.isascii() and key[:1].isalnum():
        return re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text, re.IGNORECASE) is not None
    return key in text


@dataclass(slots=True)
class Glossary:
    terms: dict[str, str] = field(default_factory=dict)
    never: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.terms or self.never)

    def relevant(self, texts: Iterable[str]) -> tuple[dict[str, str], list[str]]:
        """The terms and never-translate words that occur in `texts`, capped."""
        joined = "\n".join(texts)
        terms: dict[str, str] = {}
        used = 0
        for key in sorted(self.terms, key=len, reverse=True):
            if not _found(key, joined):
                continue
            cost = len(key) + len(self.terms[key]) + 4
            if len(terms) >= MAX_TERMS or used + cost > MAX_CHARS:
                break
            terms[key] = self.terms[key]
            used += cost
        never = [word for word in self.never if _found(word, joined)]
        return terms, never

    def prompt_section(self, texts: Iterable[str]) -> str:
        terms, never = self.relevant(texts)
        lines: list[str] = []
        if terms:
            lines.append("Glossary (always translate these terms as given):")
            lines += [f"- {src} => {dst}" for src, dst in terms.items()]
        if never:
            lines.append("Keep these exactly as written, never translate them:")
            lines += [f"- {word}" for word in never]
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str, kind: str = "auto") -> Glossary:
        stripped = text.strip()
        if kind == "json" or (kind == "auto" and stripped[:1] in "{["):
            data = json.loads(stripped)
            if isinstance(data, dict) and ("terms" in data or "never" in data):
                return cls(dict(data.get("terms", {})), list(data.get("never", [])))
            return cls(dict(data))
        glossary = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("\t", "=>", "="):
                if sep in line:
                    src, dst = line.split(sep, 1)
                    glossary.terms[src.strip()] = dst.strip()
                    break
            else:
                glossary.never.append(line)
        return glossary

    @classmethod
    def load(cls, path: str | Path) -> Glossary:
        path = Path(path)
        kind = "json" if path.suffix == ".json" else "text"
        return cls.parse(path.read_text(encoding="utf-8"), kind)

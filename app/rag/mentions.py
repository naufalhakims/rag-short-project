"""@-mentions: tagging documents in a question to search only those documents.

A mention is "@" followed by a document's file name, with or without its extension:
"@paper.pdf" or "@paper". File names may contain spaces ("@Annual Report 2024.pdf"),
so mentions are found by matching the names of the conversation's own documents,
longest name first, rather than by guessing where a mention ends.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePath

# A mention must end at the end of the text, at whitespace, or at punctuation.
_END = r"(?=$|[\s,;:!?)\]}\"'])"
_SPACES = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class Mentions:
    doc_ids: list[str] = field(default_factory=list)  # in order of first mention
    filenames: list[str] = field(default_factory=list)
    text_without: str = ""  # the question with the mentions taken out


def find_mentions(text: str, documents: Sequence[dict]) -> Mentions:
    """Find @-mentions of `documents` (dicts with "id" and "filename") in `text`."""
    names: list[tuple[str, dict]] = []
    for doc in documents:
        filename = doc["filename"]
        stem = PurePath(filename).stem
        names.append((filename, doc))
        if stem and stem != filename:
            names.append((stem, doc))
    # Longest first, so "@report 2024.pdf" wins over "@report".
    names.sort(key=lambda pair: len(pair[0]), reverse=True)

    found: list[tuple[int, dict]] = []
    masked = text
    for name, doc in names:
        pattern = re.compile("@" + re.escape(name) + _END, re.IGNORECASE)
        for match in pattern.finditer(masked):
            found.append((match.start(), doc))
        # Blank out matches so a shorter name cannot match inside a longer one.
        masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)

    if not found:
        return Mentions(text_without=text)
    seen: dict[str, str] = {}
    for _, doc in sorted(found, key=lambda pair: pair[0]):
        seen.setdefault(doc["id"], doc["filename"])
    cleaned = _SPACES.sub(" ", masked).strip()
    return Mentions(list(seen), list(seen.values()), cleaned)

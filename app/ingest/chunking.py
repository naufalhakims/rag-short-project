"""Split document text into retrieval-sized chunks.

Text is broken into small "atoms" (lines, or sentences for long lines), which are
then packed greedily into chunks of at most `size` characters. Chunks prefer to
start at Markdown headings, carry a little overlap from the previous chunk, and
remember the heading path and page they came from.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.db import NewChunk
from app.ingest.loaders import Block

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？;:])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_MAX_HEADING_PATH = 200


@dataclass(frozen=True)
class _Atom:
    text: str
    sep: str  # what joins this atom to the previous one
    page: int | None
    level: int = 0  # Markdown heading level, 0 for body text

    @property
    def size(self) -> int:
        return len(self.sep) + len(self.text)


def _split_long(text: str, size: int) -> list[str]:
    """Split a single over-long line into pieces of at most `size` characters."""
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size  # no usable space: hard cut
            pieces.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if sentence:
            pieces.append(sentence)
    return pieces


def _atoms(blocks: Sequence[Block], size: int) -> list[_Atom]:
    atoms: list[_Atom] = []
    for block in blocks:
        for paragraph in _PARAGRAPH_BREAK.split(block.text):
            sep = "\n\n"
            for line in paragraph.split("\n"):
                line = line.rstrip()
                if not line.strip():
                    continue
                heading = _HEADING.match(line)
                if heading and len(line) <= size:
                    atoms.append(_Atom(line, sep, block.page, level=len(heading.group(1))))
                elif len(line) <= size:
                    atoms.append(_Atom(line, sep, block.page))
                else:
                    for piece in _split_long(line, size):
                        atoms.append(_Atom(piece, sep, block.page))
                        sep = " "
                sep = "\n"
    return atoms


def _join(atoms: Sequence[_Atom]) -> str:
    return "".join(a.text if i == 0 else a.sep + a.text for i, a in enumerate(atoms))


def _length(atoms: Sequence[_Atom]) -> int:
    return sum(a.size for a in atoms) - (len(atoms[0].sep) if atoms else 0)


def _overlap_tail(atoms: Sequence[_Atom], overlap: int, room: int) -> list[_Atom]:
    """The trailing atoms of a finished chunk that fit in `overlap` (and in `room`)."""
    tail: list[_Atom] = []
    budget = min(overlap, room)
    for atom in reversed(atoms):
        if atom.level or _length([atom, *tail]) > budget:
            break
        tail.insert(0, atom)
    return tail


def chunk_blocks(blocks: Sequence[Block], size: int = 900, overlap: int = 120) -> list[NewChunk]:
    atoms = _atoms(blocks, size)
    chunks: list[NewChunk] = []
    headings: list[tuple[int, str]] = []  # open heading stack: (level, title)
    current: list[_Atom] = []
    carried = 0  # how many atoms at the start of `current` are overlap from the last chunk
    chunk_heading = ""

    def heading_path() -> str:
        return " > ".join(title for _, title in headings)[:_MAX_HEADING_PATH]

    def flush() -> None:
        text = _join(current).strip()
        if text:
            chunks.append(NewChunk(text=text, heading=chunk_heading, page=current[0].page))

    for atom in atoms:
        if current:
            too_big = _length(current) + atom.size > size
            # Start a new chunk at a heading once the current one has some substance.
            new_section = atom.level > 0 and _length(current) >= size * 0.4
            if too_big or new_section:
                flush()
                room = size - atom.size
                current = [] if atom.level else _overlap_tail(current, overlap, room)
                carried = len(current)
                chunk_heading = heading_path()  # section the carried-over text belongs to

        if atom.level:
            title = _HEADING.match(atom.text).group(2)
            while headings and headings[-1][0] >= atom.level:
                headings.pop()
            headings.append((atom.level, title))
        if not current:
            chunk_heading = heading_path()
        current.append(atom)

    # A tiny leftover reads better attached to the previous chunk than on its own.
    fresh = current[carried:]
    if chunks and fresh and _length(fresh) < min(120, size // 5) and not fresh[0].level:
        last = chunks[-1]
        merged = last.text + "".join(a.sep + a.text for a in fresh)
        chunks[-1] = NewChunk(text=merged, heading=last.heading, page=last.page)
    elif fresh:
        flush()
    return chunks

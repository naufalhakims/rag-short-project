"""Turn uploaded files into plain-text blocks.

Every loader returns a list of `Block`s. Headings are expressed as Markdown
(`# Title`) so the chunker can treat all formats the same way.
"""

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}


class IngestError(ValueError):
    """A problem with an uploaded file that the user can act on."""


@dataclass(frozen=True)
class Block:
    text: str
    page: int | None = None  # 1-based page number, for paginated formats


def extension_of(filename: str) -> str:
    return PurePath(filename).suffix.lower()


def load_blocks(filename: str, data: bytes) -> list[Block]:
    ext = extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestError(f"Unsupported file type '{ext or filename}'. Supported: {allowed}.")
    if ext == ".pdf":
        blocks = _load_pdf(data)
    elif ext == ".docx":
        blocks = _load_docx(data)
    else:
        blocks = [Block(_decode_text(data))]
    blocks = [Block(_clean(b.text), b.page) for b in blocks]
    blocks = [b for b in blocks if b.text]
    if not blocks:
        raise IngestError("The file contains no readable text.")
    return blocks


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:4096]:
        raise IngestError("This looks like a binary file, not text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _load_pdf(data: bytes) -> list[Block]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("This PDF is password-protected.")
        blocks = [
            Block(page.extract_text() or "", page=number)
            for number, page in enumerate(reader.pages, start=1)
        ]
    except IngestError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError) as exc:
        raise IngestError(f"Could not read this PDF ({exc.__class__.__name__}).") from exc
    if not any(b.text.strip() for b in blocks):
        raise IngestError(
            "No selectable text found in this PDF. Scanned documents need OCR first."
        )
    return blocks


def _load_docx(data: bytes) -> list[Block]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several unrelated types for bad files
        raise IngestError("Could not read this Word document.") from exc

    parts: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = (item.style.name or "") if item.style is not None else ""
            heading = re.match(r"Heading (\d)", style)
            if heading:
                parts.append(f"{'#' * int(heading.group(1))} {text}")
            elif style == "Title":
                parts.append(f"# {text}")
            elif style.startswith("List"):
                parts.append(f"- {text}")
            else:
                parts.append(text)
        elif isinstance(item, Table):
            rows = [
                " | ".join(cell.text.strip() for cell in row.cells) for row in item.rows
            ]
            parts.append("\n".join(r for r in rows if r.strip(" |")))
    return [Block("\n\n".join(parts))]

import io
from itertools import pairwise

import pytest

from app.ingest.chunking import chunk_blocks
from app.ingest.loaders import Block, IngestError, load_blocks


def chunk(text, size=200, overlap=40):
    return chunk_blocks([Block(text)], size, overlap)


# ── Chunking ─────────────────────────────────────────────────────


def test_short_text_is_one_chunk():
    chunks = chunk("A single short paragraph.")
    assert [c.text for c in chunks] == ["A single short paragraph."]


def test_chunks_respect_the_size_limit_and_lose_nothing():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(60)]
    chunks = chunk(" ".join(sentences), size=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)


def test_consecutive_chunks_overlap():
    lines = [f"Line {i} has some filler words in it." for i in range(30)]
    chunks = chunk("\n".join(lines), size=200, overlap=60)
    for previous, current in pairwise(chunks):
        first_line = current.text.split("\n")[0]
        assert first_line in previous.text  # each chunk starts with the tail of the last


def test_very_long_word_is_hard_split():
    chunks = chunk("x" * 950, size=200, overlap=0)
    assert all(len(c.text) <= 200 for c in chunks)
    assert sum(len(c.text) for c in chunks) == 950


def test_markdown_headings_start_chunks_and_are_recorded():
    body = "Details about this part of the document. " * 4
    text = f"# Guide\n\n## Rules\n{body}\n\n## Scoring\n{body}\n\n### Bonus\n{body}"
    chunks = chunk(text, size=300, overlap=0)
    headings = [c.heading for c in chunks]
    assert headings[0] == "Guide"  # the title and the first section share a chunk
    assert "Guide > Scoring" in headings
    assert "Guide > Scoring > Bonus" in headings
    scoring = next(c for c in chunks if c.heading == "Guide > Scoring")
    assert scoring.text.startswith("## Scoring")


def test_small_sections_are_merged_rather_than_fragmented():
    text = "# Title\n\n## A\nshort\n\n## B\nshort\n\n## C\nshort"
    assert len(chunk(text, size=300)) == 1


def test_pages_are_tracked():
    blocks = [Block("First page text. " * 20, page=1), Block("Second page text. " * 20, page=2)]
    chunks = chunk_blocks(blocks, 200, 0)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2


def test_empty_input_gives_no_chunks():
    assert chunk("   \n\n  ") == []


# ── Loaders ──────────────────────────────────────────────────────


def test_text_is_decoded_and_normalised():
    blocks = load_blocks("notes.txt", "﻿Line one\r\nLine two\r\n\r\n\r\n\r\nLine three".encode())
    assert blocks == [Block("Line one\nLine two\n\nLine three")]


def test_non_utf8_text_still_loads():
    assert "café" in load_blocks("legacy.txt", "café".encode("cp1252"))[0].text


def test_unsupported_and_empty_files_are_rejected():
    with pytest.raises(IngestError, match="Unsupported"):
        load_blocks("photo.png", b"\x89PNG")
    with pytest.raises(IngestError, match="no readable text"):
        load_blocks("empty.txt", b"  \n ")
    with pytest.raises(IngestError, match="binary"):
        load_blocks("fake.txt", b"MZ\x00\x00\x00binary")


def test_docx_headings_lists_and_tables():
    import docx

    document = docx.Document()
    document.add_heading("Quarterly Report", level=1)
    document.add_paragraph("Revenue grew this quarter.")
    document.add_heading("Risks", level=2)
    document.add_paragraph("Supply delays", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Region", "Sales"
    table.cell(1, 0).text, table.cell(1, 1).text = "Asia", "42"
    buffer = io.BytesIO()
    document.save(buffer)

    text = load_blocks("report.docx", buffer.getvalue())[0].text
    assert "# Quarterly Report" in text
    assert "## Risks" in text
    assert "- Supply delays" in text
    assert "Asia | 42" in text


def _minimal_pdf(pages: list[str]) -> bytes:
    """Hand-assemble a tiny text PDF (one line of Helvetica per page)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def test_pdf_text_is_extracted_per_page():
    blocks = load_blocks("paper.pdf", _minimal_pdf(["Hello from page one", "Second page here"]))
    assert [(b.page, b.text) for b in blocks] == [
        (1, "Hello from page one"),
        (2, "Second page here"),
    ]


def test_broken_pdf_is_reported_not_crashed():
    with pytest.raises(IngestError):
        load_blocks("broken.pdf", b"%PDF-1.4 this is not really a pdf")

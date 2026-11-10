"""Text extraction and chunking for DocChat.

The retriever can only find what the chunker preserves, so chunking is the
quiet workhorse of any RAG system: chunks too big drown the signal, chunks
too small lose the sentence's meaning. We split on paragraph boundaries
first (authors group related sentences into paragraphs), fall back to
sentences for oversized paragraphs, and carry a small overlap so an idea
that straddles a boundary still appears whole in at least one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    index: int
    page: int | None = None


def extract_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, raw_text)] for a PDF. Imported lazily so the
    chunker itself stays dependency-free and unit-testable."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
    page: int | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """Greedy paragraph-packing chunker with sentence fallback and overlap.

    chunk_size/overlap are in characters (roughly 4 chars per token for
    English), so 800 chars keeps every chunk well inside any embedding
    model's context window while leaving room for several retrieved chunks
    in the generation prompt.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_sentences(para))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    index = start_index

    def flush() -> None:
        nonlocal current, current_len, index
        if not current:
            return
        body = " ".join(current)
        chunks.append(Chunk(text=body, source=source, index=index, page=page))
        index += 1
        # Overlap: keep trailing units whose total length fits the overlap
        # budget, so the next chunk repeats the tail of this one.
        if overlap > 0:
            kept: list[str] = []
            kept_len = 0
            for unit in reversed(current):
                if kept_len + len(unit) + 1 > overlap:
                    break
                kept.insert(0, unit)
                kept_len += len(unit) + 1
            current = kept
            current_len = kept_len
        else:
            current = []
            current_len = 0

    for unit in units:
        if current and current_len + len(unit) + 1 > chunk_size:
            flush()
        current.append(unit)
        current_len += len(unit) + 1
        if len(unit) > chunk_size:  # single oversized sentence: hard-split
            flush()
    flush()
    return chunks


def chunk_document(
    pages: list[tuple[int, str]],
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Chunk every page of an extracted document, keeping page numbers so
    answers can cite 'page 3' back to the reader."""
    all_chunks: list[Chunk] = []
    for page_no, page_text in pages:
        page_chunks = chunk_text(
            page_text, source, chunk_size, overlap, page=page_no,
            start_index=len(all_chunks),
        )
        all_chunks.extend(page_chunks)
    return all_chunks

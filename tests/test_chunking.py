from app.chunking import chunk_document, chunk_text


def test_paragraphs_pack_into_chunks():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunk_text(text, source="doc", chunk_size=60, overlap=0)
    assert len(chunks) >= 2
    assert all(c.source == "doc" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_tail_into_next_chunk():
    units = " ".join(f"Sentence number {i}." for i in range(20))
    chunks = chunk_text(units, source="doc", chunk_size=80, overlap=30)
    assert len(chunks) > 1
    # The end of chunk N should reappear at the start of chunk N+1.
    tail = chunks[0].text.split()[-3:]
    assert " ".join(tail) in chunks[1].text


def test_oversized_sentence_is_hard_split_not_dropped():
    long_sentence = "word " * 500  # ~2500 chars, no sentence break
    chunks = chunk_text(long_sentence.strip(), source="doc", chunk_size=200, overlap=0)
    assert chunks, "an oversized sentence must still produce a chunk"
    assert "word" in chunks[0].text


def test_chunk_document_keeps_page_numbers():
    pages = [(1, "Alpha content."), (2, "Beta content.")]
    chunks = chunk_document(pages, source="manual.pdf")
    assert chunks[0].page == 1
    assert chunks[1].page == 2
    assert chunks[1].index == chunks[0].index + 1

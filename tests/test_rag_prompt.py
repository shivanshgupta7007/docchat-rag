from app.chunking import Chunk
from app.rag import build_prompt


def test_prompt_numbers_sources_and_includes_question():
    contexts = [
        Chunk(text="Refunds take 5 days.", source="policy.pdf", index=0, page=3),
        Chunk(text="Support is 24x7.", source="policy.pdf", index=1, page=4),
    ]
    prompt = build_prompt("How long do refunds take?", contexts)
    assert "[1]" in prompt and "[2]" in prompt
    assert "page 3" in prompt
    assert "How long do refunds take?" in prompt
    assert "never invent facts" in prompt.lower() or "never invent" in prompt.lower()


def test_prompt_without_page_omits_page_citation():
    contexts = [Chunk(text="Some text.", source="notes.txt", index=0, page=None)]
    prompt = build_prompt("q", contexts)
    assert "page" not in prompt.split("Question:")[0]

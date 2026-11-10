"""The RAG pipeline: retrieve -> build a grounded prompt -> generate -> cite.

Retrieval-Augmented Generation in one sentence: look up the handful of
chunks most relevant to the question and make the LLM answer using only
those, with citations, so the answer is checkable instead of plausible-
sounding.
"""
from __future__ import annotations

from .chunking import Chunk, chunk_document, extract_pdf_pages
from .embeddings import GeminiEmbedder
from .llm import GeminiChat
from .store import VectorStore

PROMPT_TEMPLATE = """You are DocChat, an assistant that answers questions about the user's documents.

Use ONLY the context below. If the context does not contain the answer, say
"I couldn't find that in the uploaded documents" - never invent facts.
Cite the chunk(s) you used as [1], [2], ... matching the context numbers.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, contexts: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        page = f", page {c.page}" if c.page else ""
        blocks.append(f"[{i}] (from {c.source}{page})\n{c.text}")
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)


class RAGPipeline:
    def __init__(
        self,
        embedder: GeminiEmbedder | None = None,
        store: VectorStore | None = None,
        llm: GeminiChat | None = None,
        top_k: int = 4,
    ):
        self.embedder = embedder or GeminiEmbedder()
        self.store = store or VectorStore()
        self.llm = llm or GeminiChat()
        self.top_k = top_k

    def ingest_pdf(self, pdf_path: str, source_name: str | None = None) -> int:
        """Extract -> chunk -> embed -> store. Returns chunks added."""
        source = source_name or pdf_path
        pages = extract_pdf_pages(pdf_path)
        chunks = chunk_document(pages, source)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def ask(self, question: str) -> dict:
        qvec = self.embedder.embed_query(question)
        hits = self.store.query(qvec, k=self.top_k)
        contexts = [c for c, _ in hits]
        answer = self.llm.generate(build_prompt(question, contexts))
        return {
            "answer": answer,
            "sources": [
                {
                    "source": c.source,
                    "page": c.page,
                    "similarity": round(score, 3),
                    "snippet": c.text[:200],
                }
                for c, score in hits
            ],
        }

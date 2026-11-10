"""Vector store: a thin wrapper over ChromaDB.

Chroma keeps the chunk vectors plus the metadata (source file, page number)
that makes citations possible. Swapping in another store (FAISS, Pinecone)
means reimplementing add() and query() - the pipeline never touches Chroma
directly.
"""
from __future__ import annotations

from .chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: str = ".chroma", collection: str = "docchat"):
        import chromadb  # lazy import: not needed for unit tests

        client = chromadb.PersistentClient(path=persist_dir)
        self._col = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._col.add(
            ids=[f"{c.source}:{c.index}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page or -1, "index": c.index} for c in chunks],
        )

    def query(self, vector: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        res = self._col.query(query_embeddings=[vector], n_results=k)
        out: list[tuple[Chunk, float]] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(
                text=doc,
                source=meta["source"],
                index=meta["index"],
                page=None if meta["page"] == -1 else meta["page"],
            )
            out.append((chunk, 1.0 - dist))  # cosine distance -> similarity
        return out

    def count(self) -> int:
        return self._col.count()

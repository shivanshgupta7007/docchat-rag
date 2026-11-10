"""Embedding providers.

An embedder turns text into a vector so 'what is the refund policy?' lands
near the paragraph that answers it, even when they share no keywords.
Any object with .embed(list[str]) -> list[list[float]] plugs into the
pipeline; Gemini's text-embedding model is the default because the project
already uses the Gemini API for generation.
"""
from __future__ import annotations

import os


class GeminiEmbedder:
    def __init__(self, api_key: str | None = None, model: str = "models/text-embedding-004"):
        import google.generativeai as genai  # lazy: heavy, only needed at runtime

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        genai.configure(api_key=key)
        self._genai = genai
        self.model = model

    def embed(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        result = self._genai.embed_content(model=self.model, content=texts, task_type=task_type)
        return result["embedding"]

    def embed_query(self, text: str) -> list[float]:
        # Retrieval works best when queries and documents are embedded with
        # their own task types - asymmetric retrieval.
        return self.embed([text], task_type="retrieval_query")[0]

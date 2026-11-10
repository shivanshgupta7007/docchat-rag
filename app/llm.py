"""Gemini chat client - the only place the project talks to the LLM,
which keeps the model name, retries and configuration in one spot."""
from __future__ import annotations

import os


class GeminiChat:
    def __init__(self, api_key: str | None = None, model: str = "gemini-3.6-flash"):
        import google.generativeai as genai  # lazy import

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)
        self.model_name = model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        # Low temperature: answers should restate the retrieved context,
        # not get creative with it.
        resp = self._model.generate_content(
            prompt, generation_config={"temperature": temperature}
        )
        return resp.text

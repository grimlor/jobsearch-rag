"""Ollama embedding wrapper with retry logic."""

from __future__ import annotations


class Embedder:
    """Wraps Ollama embedding and LLM calls with backoff and error handling."""

    def __init__(self, base_url: str, embed_model: str, llm_model: str) -> None:
        self.base_url = base_url
        self.embed_model = embed_model
        self.llm_model = llm_model

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""
        raise NotImplementedError

    async def classify(self, prompt: str) -> str:
        """Send a classification prompt to the LLM and return the raw response."""
        raise NotImplementedError

    async def health_check(self) -> None:
        """Verify Ollama is reachable and the configured models are available.

        Raises :class:`~jobsearch_rag.errors.ActionableError` on failure.
        Must be called **before** any browser session is opened.
        """
        raise NotImplementedError

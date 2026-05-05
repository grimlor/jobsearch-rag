"""Port protocol for text embedding and LLM classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jobsearch_rag.models import InferenceMetrics


@runtime_checkable
class EmbedderPort(Protocol):
    """Port protocol for text embedding and LLM classification."""

    max_embed_chars: int
    llm_model: str

    @property
    def metrics(self) -> InferenceMetrics:
        """Accumulated inference metrics for the current session."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""
        ...

    async def classify(self, prompt: str) -> str:
        """Send a classification prompt and return the raw response."""
        ...

    async def health_check(self) -> None:
        """Verify the embedding backend is reachable and ready."""
        ...

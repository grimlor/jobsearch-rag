"""
Test fakes for port protocols.

Provides deterministic, configurable test doubles that satisfy port
protocols without depending on external services.

Classes:
    FakeEmbedder — satisfies :class:`~jobsearch_rag.ports.EmbeddingPort`
        (but NOT HealthCheckable or MetricsProvider).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeEmbedder:
    """
    Deterministic test double satisfying :class:`EmbeddingPort`.

    Does **not** satisfy :class:`HealthCheckable` or
    :class:`MetricsProvider` — validates the ``isinstance`` guard
    pattern in :class:`PipelineRunner`.

    Parameters
    ----------
    embed_vector:
        Fixed vector returned by :meth:`embed` (default all-zeros length 8).
    classify_response:
        Fixed string returned by :meth:`classify` (default ``"{}"``).
    embed_side_effect:
        Optional callable ``(text) -> list[float]`` that overrides
        *embed_vector* when set.
    classify_side_effect:
        Optional callable ``(prompt) -> str`` that overrides
        *classify_response* when set.

    """

    def __init__(
        self,
        *,
        embed_vector: list[float] | None = None,
        classify_response: str = "{}",
        embed_side_effect: Callable[[str], list[float]] | None = None,
        classify_side_effect: Callable[[str], str] | None = None,
    ) -> None:
        """Initialise the fake with configurable return values and side effects."""
        self._embed_vector = embed_vector if embed_vector is not None else [0.0] * 8
        self._classify_response = classify_response
        self._embed_side_effect = embed_side_effect
        self._classify_side_effect = classify_side_effect
        self.embed_calls: list[str] = []
        self.embed_call_count: int = 0
        self.classify_call_count: int = 0

    async def embed(self, text: str) -> list[float]:
        """Return the configured vector (or side_effect result) for *text*."""
        self.embed_calls.append(text)
        self.embed_call_count += 1
        if self._embed_side_effect is not None:
            return self._embed_side_effect(text)
        return list(self._embed_vector)

    async def classify(self, prompt: str) -> str:
        """Return the configured response (or side_effect result) for *prompt*."""
        self.classify_call_count += 1
        if self._classify_side_effect is not None:
            return self._classify_side_effect(prompt)
        return self._classify_response

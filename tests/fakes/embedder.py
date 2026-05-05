"""FakeEmbedder — deterministic, zero-I/O implementation of EmbedderPort."""

from __future__ import annotations

import math

from jobsearch_rag.models import InferenceMetrics

_DEFAULT_EMBED_VECTOR: list[float] = [1.0, 0.0, 0.0]
_DEFAULT_MAX_EMBED_CHARS: int = 8192


class FakeEmbedder:
    """Deterministic, zero-I/O implementation of EmbedderPort for testing."""

    def __init__(
        self,
        *,
        embed_vector: list[float] | None = None,
        max_embed_chars: int = _DEFAULT_MAX_EMBED_CHARS,
        classify_response: str = '{"disqualified": false}',
        health_check_error: Exception | None = None,
        above_threshold_count: int = 0,
        similarity: float | None = None,
    ) -> None:
        """Configure deterministic embedding behavior via constructor params."""
        if above_threshold_count > 0 and similarity is not None:
            msg = (
                "Cannot set both above_threshold_count and similarity — "
                "they control embed() output through conflicting strategies"
            )
            raise ValueError(msg)
        self._embed_vector = (
            embed_vector if embed_vector is not None else list(_DEFAULT_EMBED_VECTOR)
        )
        self.max_embed_chars = max_embed_chars
        self.llm_model = "fake-llm"
        self._classify_response = classify_response
        self._health_check_error = health_check_error
        self._above_threshold_count = above_threshold_count
        self._similarity = similarity
        self._embed_call_count = 0

    @property
    def metrics(self) -> InferenceMetrics:
        """Return zero-valued metrics — protocol conformance."""
        return InferenceMetrics()

    async def embed(self, text: str) -> list[float]:
        """Return a deterministic embedding vector controlled by constructor params."""
        if self._above_threshold_count > 0:
            self._embed_call_count += 1
            if self._embed_call_count <= self._above_threshold_count:
                return list(self._embed_vector)
            return _orthogonal_vector(self._embed_vector)

        if self._similarity is not None:
            return _vector_with_similarity(self._embed_vector, self._similarity)

        return list(self._embed_vector)

    async def classify(self, prompt: str) -> str:
        """Return the configured classify_response unconditionally."""
        return self._classify_response

    async def health_check(self) -> None:
        """Raise health_check_error if configured; otherwise no-op."""
        if self._health_check_error is not None:
            raise self._health_check_error


def _vector_with_similarity(reference: list[float], sim: float) -> list[float]:
    """Return a unit vector with cosine similarity *sim* to *reference*."""
    dim = len(reference)
    norm = math.sqrt(sum(x * x for x in reference))
    if norm == 0:
        return [0.0] * dim
    ref = [x / norm for x in reference]

    # Construct a vector orthogonal to ref
    ortho = _raw_orthogonal(ref)
    ortho_norm = math.sqrt(sum(x * x for x in ortho))
    if ortho_norm > 0:
        ortho = [x / ortho_norm for x in ortho]

    complement = math.sqrt(max(0.0, 1.0 - sim * sim))
    return [sim * ref[i] + complement * ortho[i] for i in range(dim)]


def _orthogonal_vector(reference: list[float]) -> list[float]:
    """Return a vector orthogonal to *reference* (cosine similarity ≈ 0)."""
    return _vector_with_similarity(reference, 0.0)


def _raw_orthogonal(unit_ref: list[float]) -> list[float]:
    """Construct a non-zero vector orthogonal to the given unit vector."""
    dim = len(unit_ref)
    ortho = [0.0] * dim
    # Find first two indices where ref has value — rotate in that plane
    for i in range(dim):
        if unit_ref[i] != 0.0:
            j = (i + 1) % dim
            ortho[i] = -unit_ref[j]
            ortho[j] = unit_ref[i]
            return ortho
    return ortho

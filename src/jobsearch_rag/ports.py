"""
Hexagonal port interfaces for the domain boundary.

Defines :class:`Protocol` interfaces that decouple domain classes
(Scorer, DecisionRecorder, Indexer, EvalRunner, PipelineRunner)
from concrete infrastructure (Ollama SDK, ChromaDB).  Any
implementation satisfying the protocol can be injected — including
test fakes.

Protocols:
    EmbeddingPort — core embedding and LLM classification operations.
    HealthCheckable — supplementary pre-flight connectivity check.
    MetricsProvider — supplementary inference metrics exposure.
    VectorStorePort — all vector storage operations.

Result dataclasses:
    QueryResult — typed replacement for dict[str, Any] query returns.
    GetResult — typed replacement for dict[str, Any] get returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jobsearch_rag.rag.embedder import InferenceMetrics

# ============================================================================
# Result Dataclasses
# ============================================================================


@dataclass
class QueryResult:
    """
    Typed result from :meth:`VectorStorePort.query`.

    Batch shape (nested lists) matches ChromaDB's native ``query()``
    response — one inner list per query embedding.
    """

    ids: list[list[str]] = field(default_factory=lambda: [[]])
    documents: list[list[str]] = field(default_factory=lambda: [[]])
    metadatas: list[list[dict[str, Any]]] = field(default_factory=lambda: [[]])
    distances: list[list[float]] = field(default_factory=lambda: [[]])


@dataclass
class GetResult:
    """
    Typed result from :meth:`VectorStorePort.get_documents` and friends.

    Flat shape (single list) matches ChromaDB's native ``get()``
    response.
    """

    ids: list[str] = field(default_factory=lambda: list[str]())
    documents: list[str | None] = field(default_factory=lambda: list[str | None]())
    metadatas: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())


# ============================================================================
# Embedding Port
# ============================================================================


@runtime_checkable
class EmbeddingPort(Protocol):
    """
    Core embedding and LLM classification operations.

    Domain classes depend on this protocol instead of the concrete
    :class:`~jobsearch_rag.rag.embedder.Embedder`.  Keeping the port
    narrow (embed + classify only) means test fakes don't need
    health-check or metrics stubs.
    """

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""
        ...

    async def classify(self, prompt: str) -> str:
        """Return the LLM classification response for *prompt*."""
        ...


# ============================================================================
# Supplementary Observability Protocols
# ============================================================================


@runtime_checkable
class HealthCheckable(Protocol):
    """
    Pre-flight connectivity verification.

    :class:`PipelineRunner` uses an ``isinstance`` guard to optionally
    call :meth:`health_check` before pipeline execution.  Implementations
    that do not support health checks (e.g. test fakes) simply don't
    satisfy this protocol — the guard is skipped silently.
    """

    async def health_check(self) -> None:
        """Verify connectivity to the underlying service."""
        ...


@runtime_checkable
class MetricsProvider(Protocol):
    """
    Inference metrics exposure.

    :class:`PipelineRunner` uses an ``isinstance`` guard to optionally
    collect :attr:`metrics` during the session summary.
    """

    @property
    def metrics(self) -> InferenceMetrics:
        """Accumulated inference metrics for the current session."""
        ...


# ============================================================================
# Vector Store Port
# ============================================================================


@runtime_checkable
class VectorStorePort(Protocol):
    """
    All vector storage operations the domain requires.

    Domain classes depend on this protocol instead of the concrete
    :class:`~jobsearch_rag.rag.store.VectorStore`.  Implementations
    include the real ChromaDB-backed store and an in-memory test fake.
    """

    def add_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add (or upsert) documents with pre-computed embeddings."""
        ...

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> QueryResult:
        """Find the *n_results* most similar documents to *query_embedding*."""
        ...

    def get_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> GetResult:
        """Retrieve documents by ID from a collection."""
        ...

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: dict[str, Any],
        include: list[str] | None = None,
    ) -> GetResult:
        """Retrieve documents matching a metadata filter."""
        ...

    def get_all_documents(
        self,
        collection_name: str,
        *,
        include: list[str] | None = None,
    ) -> GetResult:
        """Retrieve all documents in a collection."""
        ...

    def delete_by_id(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """Delete documents by ID from a collection."""
        ...

    def collection_count(self, name: str) -> int:
        """Return the document count for collection *name*."""
        ...

    def reset_collection(self, name: str) -> None:
        """Drop and recreate the named collection (empty)."""
        ...

    def close(self) -> None:
        """Release resources held by the store."""
        ...

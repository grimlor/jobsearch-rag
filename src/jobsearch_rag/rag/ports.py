"""
Hexagonal port for vector storage.

Defines the domain types and protocol that callers depend on.
The concrete adapter (ChromaVectorStore in store.py) implements this protocol.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from types import TracebackType


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """A single document retrieved from the vector store."""

    id: str
    document: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    """A document with its embedding, ready for storage."""

    id: str
    document: str
    embedding: list[float]
    metadata: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ScoredMatch:
    """A single similarity match with its distance from the query vector."""

    id: str
    document: str
    metadata: dict[str, str]
    distance: float


@dataclass(frozen=True, slots=True)
class QueryResults:
    """Results from a similarity query — flat list, not nested."""

    matches: list[ScoredMatch]


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Typed domain filter for metadata queries."""

    field: str
    operator: Literal["eq", "ne"]
    value: str


# ---------------------------------------------------------------------------
# Port protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStorePort(Protocol):
    """What the domain needs from a vector store."""

    def close(self) -> None:
        """Release resources held by the store."""
        ...

    def __enter__(self) -> Self:
        """Enter the runtime context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context, releasing resources."""
        ...

    def collection_count(self, name: str) -> int:
        """Return the number of documents in the named collection."""
        ...

    def reset_collection(self, name: str) -> None:
        """Drop and recreate the named collection, removing all documents."""
        ...

    def add_documents(
        self,
        collection_name: str,
        *,
        documents: list[EmbeddedDocument],
    ) -> None:
        """Store documents with embeddings, upserting by ID."""
        ...

    def get_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> list[DocumentRecord]:
        """Retrieve documents by their IDs."""
        ...

    def get_all_documents(
        self,
        collection_name: str,
    ) -> list[DocumentRecord]:
        """Retrieve all documents in the named collection."""
        ...

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: MetadataFilter,
    ) -> list[DocumentRecord]:
        """Retrieve documents matching a metadata filter."""
        ...

    def delete_by_id(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """Remove documents by their IDs."""
        ...

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int,
    ) -> QueryResults:
        """Find the n closest documents to the query embedding."""
        ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """Configuration for constructing a VectorStorePort implementation."""

    persist_dir: str
    distance_metric: str
    sync_threshold: int
    store_class: str = "jobsearch_rag.rag.store.ChromaVectorStore"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vector_store(config: VectorStoreConfig, **kwargs: Any) -> VectorStorePort:
    """Instantiate a VectorStorePort from config via reflection."""
    module_path, class_name = config.store_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    store = cls(
        persist_dir=config.persist_dir,
        distance_metric=config.distance_metric,
        sync_threshold=config.sync_threshold,
        **kwargs,
    )
    if not isinstance(store, VectorStorePort):
        raise TypeError(f"{config.store_class} does not implement VectorStorePort")
    return store

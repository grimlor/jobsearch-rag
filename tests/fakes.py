"""
In-memory VectorStorePort implementation for unit tests.

FakeVectorStore provides deterministic, fast execution with no I/O.
It validates that domain callers depend only on the port, not ChromaDB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from types import TracebackType

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.ports import (
    DocumentRecord,
    EmbeddedDocument,
    MetadataFilter,
    QueryResults,
    ScoredMatch,
    VectorStorePort,
)


@dataclass
class _StoredDocument:
    """Internal storage representation."""

    id: str
    document: str
    embedding: list[float]
    metadata: dict[str, str]


class FakeVectorStore:
    """
    In-memory VectorStorePort for unit tests.

    No I/O, no cleanup needed. Context manager is a no-op.
    Accepts **kwargs in __init__ so the factory can pass persist_dir etc.

    Optional failure injection:
        fail_on_collections: a set of collection names for which
        get_all_documents will raise RuntimeError.
    """

    def __init__(self, *, fail_on_collections: set[str] | None = None, **_kwargs: Any) -> None:
        """Initialize with empty collections; kwargs are accepted and ignored."""
        self._collections: dict[str, list[_StoredDocument]] = {}
        self._fail_on_collections: set[str] = fail_on_collections or set()

    # -- Context manager (no-op) ---------------------------------------------

    def close(self) -> None:
        """No-op — nothing to release."""

    def __enter__(self) -> Self:
        """Return self for context-manager usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """No-op exit."""

    # -- Collection lifecycle ------------------------------------------------

    def collection_count(self, name: str) -> int:
        """Return document count for the named collection."""
        if name not in self._collections:
            raise ActionableError(
                error_type=ErrorType.INDEX,
                error=f"Collection '{name}' does not exist",
                service="vector_store",
                suggestion=f"Run indexing to create the '{name}' collection.",
            )
        return len(self._collections[name])

    def reset_collection(self, name: str) -> None:
        """Drop all documents in the named collection."""
        self._collections[name] = []

    # -- Document operations -------------------------------------------------

    def add_documents(
        self,
        collection_name: str,
        *,
        documents: list[EmbeddedDocument],
    ) -> None:
        """Store documents with upsert semantics (replace existing by ID)."""
        collection = self._collections.setdefault(collection_name, [])
        for doc in documents:
            # Upsert: remove existing with same ID, then append
            collection[:] = [d for d in collection if d.id != doc.id]
            collection.append(
                _StoredDocument(
                    id=doc.id,
                    document=doc.document,
                    embedding=doc.embedding,
                    metadata=doc.metadata or {},
                )
            )

    def get_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> list[DocumentRecord]:
        """Retrieve documents by ID."""
        if collection_name not in self._collections:
            raise ActionableError(
                error_type=ErrorType.INDEX,
                error=f"Collection '{collection_name}' does not exist",
                service="vector_store",
                suggestion=f"Run indexing to create the '{collection_name}' collection.",
            )
        collection = self._collections[collection_name]
        id_set = set(ids)
        return [
            DocumentRecord(id=d.id, document=d.document, metadata=d.metadata)
            for d in collection
            if d.id in id_set
        ]

    def get_all_documents(
        self,
        collection_name: str,
    ) -> list[DocumentRecord]:
        """Retrieve every document in the named collection."""
        if collection_name in self._fail_on_collections:
            raise RuntimeError(f"simulated collection failure: {collection_name}")
        collection = self._collections.get(collection_name, [])
        return [
            DocumentRecord(id=d.id, document=d.document, metadata=d.metadata) for d in collection
        ]

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: MetadataFilter,
    ) -> list[DocumentRecord]:
        """Filter documents by a single metadata predicate."""
        if collection_name not in self._collections:
            raise ActionableError(
                error_type=ErrorType.INDEX,
                error=f"Collection '{collection_name}' does not exist",
                service="vector_store",
                suggestion=f"Run indexing to create the '{collection_name}' collection.",
            )
        collection = self._collections[collection_name]
        results: list[DocumentRecord] = []
        for d in collection:
            value = d.metadata.get(where.field)
            if (where.operator == "eq" and value == where.value) or (
                where.operator == "ne" and value != where.value
            ):
                results.append(DocumentRecord(id=d.id, document=d.document, metadata=d.metadata))
        return results

    def delete_by_id(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """Remove documents by ID from the named collection."""
        collection = self._collections.get(collection_name, [])
        id_set = set(ids)
        self._collections[collection_name] = [d for d in collection if d.id not in id_set]

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int,
    ) -> QueryResults:
        """Return the n closest documents by cosine distance."""
        collection = self._collections.get(collection_name, [])
        if not collection:
            return QueryResults(matches=[])

        scored: list[tuple[float, _StoredDocument]] = []
        for d in collection:
            dist = _cosine_distance(query_embedding, d.embedding)
            scored.append((dist, d))

        scored.sort(key=lambda x: x[0])
        top_n: list[tuple[float, _StoredDocument]] = scored[:n_results]

        return QueryResults(
            matches=[
                ScoredMatch(
                    id=d.id,
                    document=d.document,
                    metadata=d.metadata,
                    distance=dist,
                )
                for dist, d in top_n
            ]
        )


# Ensure FakeVectorStore satisfies the protocol at import time
_: type[VectorStorePort] = FakeVectorStore


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))

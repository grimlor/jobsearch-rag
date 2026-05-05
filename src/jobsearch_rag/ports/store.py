"""Port protocol for vector document storage and similarity queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jobsearch_rag.models import DocumentRecord, QueryResult


@runtime_checkable
class VectorStorePort(Protocol):
    """Port protocol for vector document storage and similarity queries."""

    def close(self) -> None:
        """Release resources held by the store implementation."""
        ...

    def collection_count(self, name: str) -> int:
        """Return the document count for collection *name*."""
        ...

    def reset_collection(self, name: str) -> None:
        """Drop and recreate the named collection."""
        ...

    def add_documents(
        self,
        collection_name: str,
        *,
        documents: list[DocumentRecord],
    ) -> None:
        """Persist documents into the named collection."""
        ...

    def get_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> list[DocumentRecord]:
        """Retrieve documents by ID from a collection."""
        ...

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: dict[str, Any],
    ) -> list[DocumentRecord]:
        """Retrieve documents matching a metadata filter."""
        ...

    def delete_by_id(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """Delete documents by ID from a collection."""
        ...

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int,
    ) -> QueryResult:
        """Find the *n_results* most similar documents to *query_embedding*."""
        ...

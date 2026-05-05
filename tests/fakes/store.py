"""FakeVectorStore — in-memory dict-backed implementation of VectorStorePort."""

from __future__ import annotations

import math
from typing import Any

from jobsearch_rag.models import DocumentRecord, QueryMatch, QueryResult


class FakeVectorStore:
    """In-memory dict-backed implementation of VectorStorePort for testing."""

    def __init__(self) -> None:
        """Initialize with empty in-memory collections."""
        self._collections: dict[str, list[DocumentRecord]] = {}

    def close(self) -> None:
        """No-op — no resources to release in-memory."""

    def collection_count(self, name: str) -> int:
        """Return document count for collection *name*."""
        return len(self._collections.get(name, []))

    def reset_collection(self, name: str) -> None:
        """Clear all documents in the named collection."""
        self._collections[name] = []

    def add_documents(
        self,
        collection_name: str,
        *,
        documents: list[DocumentRecord],
    ) -> None:
        """Persist documents into the named collection."""
        for doc in documents:
            if doc.embedding is None:
                msg = (
                    f"DocumentRecord '{doc.id}' has embedding=None — "
                    f"embeddings must be computed before storing"
                )
                raise ValueError(msg)
        if collection_name not in self._collections:
            self._collections[collection_name] = []
        existing = self._collections[collection_name]
        new_ids = {doc.id for doc in documents}
        self._collections[collection_name] = [d for d in existing if d.id not in new_ids]
        self._collections[collection_name].extend(documents)

    def get_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> list[DocumentRecord]:
        """Retrieve documents by ID from a collection."""
        id_set = set(ids)
        return [d for d in self._collections.get(collection_name, []) if d.id in id_set]

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: dict[str, Any],
    ) -> list[DocumentRecord]:
        """Filter stored documents on the metadata predicate."""
        results: list[DocumentRecord] = []
        for doc in self._collections.get(collection_name, []):
            if doc.metadata is None:
                continue
            if all(doc.metadata.get(k) == v for k, v in where.items()):
                results.append(doc)
        return results

    def delete_by_id(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """Remove documents by ID from a collection."""
        if collection_name in self._collections:
            id_set = set(ids)
            self._collections[collection_name] = [
                d for d in self._collections[collection_name] if d.id not in id_set
            ]

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int,
    ) -> QueryResult:
        """Compute cosine distance and return top-N matches."""
        docs = self._collections.get(collection_name, [])
        scored: list[tuple[float, DocumentRecord]] = []
        for doc in docs:
            if doc.embedding is None:
                continue
            distance = _cosine_distance(query_embedding, doc.embedding)
            scored.append((distance, doc))

        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]
        matches = [
            QueryMatch(
                id=doc.id,
                document=doc.document,
                distance=dist,
                metadata=doc.metadata,
            )
            for dist, doc in top
        ]
        return QueryResult(matches=matches)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return 1.0 - similarity

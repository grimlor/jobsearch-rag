"""
Test fakes for port protocols.

Provides deterministic, configurable test doubles that satisfy port
protocols without depending on external services.

Classes:
    FakeEmbedder -- satisfies :class:`~jobsearch_rag.ports.EmbeddingPort`
        (but NOT HealthCheckable or MetricsProvider).
    InMemoryVectorStore -- satisfies :class:`~jobsearch_rag.ports.VectorStorePort`
        with dict-backed storage and cosine similarity. No ChromaDB dependency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.ports import GetResult, QueryResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from jobsearch_rag.config import Settings


class FakeEmbedder:
    """
    Deterministic test double satisfying :class:`EmbeddingPort`.

    Does **not** satisfy :class:`HealthCheckable` or
    :class:`MetricsProvider` -- validates the ``isinstance`` guard
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
        max_embed_chars: int = 8192,
        llm_model: str = "fake-model",
    ) -> None:
        """Initialise the fake with configurable return values and side effects."""
        self.embed_vector = embed_vector if embed_vector is not None else [0.0] * 8
        self.classify_response = classify_response
        self.embed_side_effect = embed_side_effect
        self.classify_side_effect = classify_side_effect
        self.embed_calls: list[str] = []
        self.classify_calls: list[str] = []
        self.embed_call_count: int = 0
        self.classify_call_count: int = 0
        self.max_embed_chars: int = max_embed_chars
        self.llm_model: str = llm_model

    async def embed(self, text: str) -> list[float]:
        """Return the configured vector (or side_effect result) for *text*."""
        self.embed_calls.append(text)
        self.embed_call_count += 1
        if self.embed_side_effect is not None:
            return self.embed_side_effect(text)
        return list(self.embed_vector)

    async def classify(self, prompt: str) -> str:
        """Return the configured response (or side_effect result) for *prompt*."""
        self.classify_calls.append(prompt)
        self.classify_call_count += 1
        if self.classify_side_effect is not None:
            return self.classify_side_effect(prompt)
        return self.classify_response

    @classmethod
    def from_settings(cls, settings: Settings) -> FakeEmbedder:
        """Construct a default FakeEmbedder (ignores *settings*)."""
        return cls()


# ============================================================================
# InMemoryVectorStore
# ============================================================================

# Internal per-document record
_DocRecord = dict[str, Any]  # keys: document, embedding, metadata


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """
    Dict-backed test double satisfying :class:`VectorStorePort`.

    Provides per-collection storage with cosine similarity search.
    No ChromaDB dependency, no temp directories, no WAL isolation.
    Writes are immediately visible from all references.
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory store."""
        # collection_name -> {doc_id -> _DocRecord}
        self._collections: dict[str, dict[str, _DocRecord]] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> InMemoryVectorStore:
        """Construct an empty InMemoryVectorStore (ignores *settings*)."""
        return cls()

    def get_or_create_collection(self, name: str) -> dict[str, _DocRecord]:
        """Ensure a collection exists (creating an empty one if needed) and return it."""
        return self._collections.setdefault(name, {})

    def _require_collection(self, name: str) -> dict[str, _DocRecord]:
        """Return the collection dict, raising ActionableError if absent."""
        if name not in self._collections:
            raise ActionableError.index(name)
        return self._collections[name]

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
        coll = self._collections.setdefault(collection_name, {})
        for i, doc_id in enumerate(ids):
            coll[doc_id] = {
                "document": documents[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i] if metadatas else {},
            }

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> QueryResult:
        """Find the *n_results* most similar documents by cosine distance."""
        coll = self._require_collection(collection_name)
        if not coll:
            return QueryResult()

        scored: list[tuple[str, float, str, dict[str, Any] | None]] = []
        for doc_id, rec in coll.items():
            dist = _cosine_distance(query_embedding, rec["embedding"])
            scored.append((doc_id, dist, rec["document"], rec.get("metadata")))

        scored.sort(key=lambda t: t[1])
        scored = scored[:n_results]

        return QueryResult(
            ids=[[s[0] for s in scored]],
            documents=[[s[2] for s in scored]],
            metadatas=[[s[3] for s in scored]],
            distances=[[s[1] for s in scored]],
        )

    def get_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> GetResult:
        """Retrieve documents by ID from a collection."""
        coll = self._require_collection(collection_name)
        out_ids: list[str] = []
        out_docs: list[str | None] = []
        out_metas: list[dict[str, Any]] = []
        for doc_id in ids:
            if doc_id in coll:
                rec = coll[doc_id]
                out_ids.append(doc_id)
                out_docs.append(rec["document"])
                out_metas.append(rec.get("metadata") or {})
        return GetResult(ids=out_ids, documents=out_docs, metadatas=out_metas)

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: dict[str, Any],
        include: list[str] | None = None,
    ) -> GetResult:
        """Retrieve documents matching a metadata filter."""
        coll = self._require_collection(collection_name)
        out_ids: list[str] = []
        out_docs: list[str | None] = []
        out_metas: list[dict[str, Any]] = []
        for doc_id, rec in coll.items():
            meta: dict[str, Any] = rec.get("metadata") or {}
            if all(meta.get(k) == v for k, v in where.items()):
                out_ids.append(doc_id)
                out_docs.append(rec["document"])
                out_metas.append(meta)
        return GetResult(ids=out_ids, documents=out_docs, metadatas=out_metas)

    def get_all_documents(
        self,
        collection_name: str,
        *,
        include: list[str] | None = None,
    ) -> GetResult:
        """Retrieve all documents in a collection."""
        coll = self._require_collection(collection_name)
        out_ids: list[str] = []
        out_docs: list[str | None] = []
        out_metas: list[dict[str, Any]] = []
        for doc_id, rec in coll.items():
            out_ids.append(doc_id)
            out_docs.append(rec["document"])
            out_metas.append(rec.get("metadata") or {})
        return GetResult(ids=out_ids, documents=out_docs, metadatas=out_metas)

    def delete_by_id(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """Delete documents by ID from a collection."""
        coll = self._require_collection(collection_name)
        for doc_id in ids:
            coll.pop(doc_id, None)

    def collection_count(self, name: str) -> int:
        """Return the document count for collection *name*, raising if absent."""
        coll = self._require_collection(name)
        return len(coll)

    def reset_collection(self, name: str) -> None:
        """Drop all documents in the named collection."""
        self._collections[name] = {}

    def close(self) -> None:
        """No-op -- satisfies the protocol."""

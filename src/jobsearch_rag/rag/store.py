"""
ChromaDB collection management.

Provides a thin wrapper around ChromaDB's client, adding:
- Consistent error handling via ActionableError
- Collection lifecycle (create, count, reset, query)
- Input validation for document operations

ChromaDB is an **embedded** vector database — like SQLite for vectors.
It stores documents alongside their embedding vectors and supports
similarity queries: "give me the N documents most similar to this vector."

Three collections serve distinct scoring purposes:

  - ``resume``         — resume chunks for fit_score
  - ``role_archetypes`` — ideal role descriptions for archetype_score
  - ``decisions``       — past accept/reject choices for history_score
"""

from __future__ import annotations

from typing import Any

import chromadb

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.logging import logger
from jobsearch_rag.models import DocumentRecord, QueryMatch, QueryResult


class VectorStore:
    """
    Manages ChromaDB collections for resume, archetypes, and decisions.

    Usage::

        store = VectorStore(persist_dir="./data/chroma_db")
        store.get_or_create_collection("resume")
        store.add_documents("resume", ids=[...], documents=[...], embeddings=[...])
        results = store.query("resume", query_embedding=[...], n_results=5)
    """

    def __init__(self, persist_dir: str, distance_metric: str) -> None:
        """Initialize ChromaDB client at *persist_dir*."""
        self.persist_dir = persist_dir
        self._distance_metric = distance_metric
        self._client = chromadb.PersistentClient(path=persist_dir)
        logger.debug("ChromaDB client initialized at %s", persist_dir)

    def close(self) -> None:
        """
        Release ChromaDB file handles (SQLite, HNSW segments).

        Must be called before deleting the *persist_dir* on Windows,
        where POSIX unlink-while-open semantics are unavailable.
        """
        self._client.clear_system_cache()

    # -- Collection lifecycle ------------------------------------------------

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Return the named ChromaDB collection, creating if necessary.

        Uses cosine similarity as the distance function — the natural
        choice for comparing text embeddings.
        """
        collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": self._distance_metric},
        )
        logger.debug("Collection '%s' ready (%d documents)", name, collection.count())
        return collection

    def collection_count(self, name: str) -> int:
        """
        Return the document count for collection *name*.

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(name)
        return collection.count()

    def reset_collection(self, name: str) -> None:
        """
        Drop and recreate the named collection.

        Safe to call on nonexistent collections (no-op for the delete,
        but always ensures the collection exists and is empty afterward).
        """
        try:
            self._client.delete_collection(name)
            logger.info("Collection '%s' deleted", name)
        except chromadb.errors.NotFoundError:
            logger.debug("Collection '%s' does not exist — nothing to reset", name)
        # Recreate empty so callers can immediately use the collection
        self.get_or_create_collection(name)
        logger.debug("Collection '%s' recreated empty", name)

    # -- Document operations -------------------------------------------------

    def add_documents(
        self,
        collection_name: str,
        *,
        documents: list[DocumentRecord],
    ) -> None:
        """
        Add (or update) documents with pre-computed embeddings.

        Documents with existing IDs are **upserted** (updated in place).

        Args:
            collection_name: Target collection (created if absent).
            documents: Document records with id, document text, embedding,
                and optional metadata.

        """
        ids = [d.id for d in documents]
        texts = [d.document for d in documents]
        embeddings = [d.embedding for d in documents if d.embedding is not None]
        if len(embeddings) != len(documents):
            msg = "All documents must have non-None embeddings for storage"
            raise ActionableError.validation(
                field_name="embeddings",
                reason=msg,
                suggestion="Compute embeddings before calling add_documents",
            )
        metadatas = [d.metadata for d in documents if d.metadata is not None]
        metadatas_arg: list[dict[str, Any]] | None = (
            metadatas if len(metadatas) == len(documents) else None
        )

        collection = self.get_or_create_collection(collection_name)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas_arg,
        )
        logger.info(
            "Upserted %d documents into '%s' (total: %d)",
            len(ids),
            collection_name,
            collection.count(),
        )

    def get_documents(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> list[DocumentRecord]:
        """
        Retrieve documents by ID from a collection.

        Returns a list of DocumentRecord instances matching the requested
        ids. Documents not found are omitted from the result.

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)
        result = collection.get(ids=ids, include=["documents", "metadatas"])
        return _chroma_get_to_records(result)

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: dict[str, Any],
    ) -> list[DocumentRecord]:
        """
        Retrieve documents matching a metadata filter.

        Uses ChromaDB's ``where`` filter syntax (e.g.
        ``{"verdict": "no"}``).

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)
        result = collection.get(where=where, include=["documents", "metadatas"])
        return _chroma_get_to_records(result)

    # -- Similarity query ----------------------------------------------------

    def delete_by_id(
        self,
        collection_name: str,
        *,
        ids: list[str],
    ) -> None:
        """
        Delete documents by ID from a collection.

        Silently ignores IDs that do not exist in the collection.

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)
        collection.delete(ids=ids)
        logger.info(
            "Deleted %d document(s) from '%s' (total: %d)",
            len(ids),
            collection_name,
            collection.count(),
        )

    # -- Similarity query (continued) ----------------------------------------

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int,
    ) -> QueryResult:
        """
        Find the *n_results* most similar documents to *query_embedding*.

        Returns a QueryResult with matches ordered by ascending distance
        (lower = more similar; 0.0 = identical direction).

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)

        # ChromaDB raises if n_results > count; clamp to available
        count = collection.count()
        if count == 0:
            return QueryResult(matches=[])

        effective_n = min(n_results, count)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_n,
            include=["documents", "metadatas", "distances"],
        )
        return _chroma_query_to_result(result)

    # -- Internal helpers ----------------------------------------------------

    def _get_existing_collection(self, name: str) -> chromadb.Collection:
        """
        Retrieve a collection that must already exist.

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection has not been created.
        """
        try:
            return self._client.get_collection(name)
        except chromadb.errors.NotFoundError:
            raise ActionableError.index(name) from None


# -- ChromaDB dict → typed model converters ----------------------------------


def _chroma_get_to_records(result: dict[str, Any]) -> list[DocumentRecord]:
    """Convert a ChromaDB ``get()`` result dict to a list of DocumentRecord."""
    ids: list[str] = result.get("ids", [])
    documents: list[str | None] = result.get("documents", [])
    metadatas: list[dict[str, Any] | None] = result.get("metadatas", [])
    records: list[DocumentRecord] = []
    for i, doc_id in enumerate(ids):
        doc_text = documents[i] if i < len(documents) else None
        meta = metadatas[i] if i < len(metadatas) else None
        records.append(
            DocumentRecord(
                id=doc_id,
                document=doc_text or "",
                metadata=meta,
            )
        )
    return records


def _chroma_query_to_result(result: dict[str, Any]) -> QueryResult:
    """Convert a ChromaDB ``query()`` result dict to a QueryResult."""
    ids_lists: list[list[str]] = result.get("ids", [[]])
    doc_lists: list[list[str | None]] = result.get("documents", [[]])
    meta_lists: list[list[dict[str, Any] | None]] = result.get("metadatas", [[]])
    dist_lists: list[list[float]] = result.get("distances", [[]])

    ids = ids_lists[0] if ids_lists else []
    documents = doc_lists[0] if doc_lists else []
    metadatas = meta_lists[0] if meta_lists else []
    distances = dist_lists[0] if dist_lists else []

    matches: list[QueryMatch] = []
    for i, doc_id in enumerate(ids):
        matches.append(
            QueryMatch(
                id=doc_id,
                document=(documents[i] if i < len(documents) else None) or "",
                distance=distances[i] if i < len(distances) else 1.0,
                metadata=metadatas[i] if i < len(metadatas) else None,
            )
        )
    return QueryResult(matches=matches)

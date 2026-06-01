"""
Concrete VectorStorePort adapter backed by ChromaDB.

See :mod:`jobsearch_rag.rag.ports` for the protocol definition and factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import chromadb

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.logging import logger
from jobsearch_rag.rag.ports import (
    DocumentRecord,
    EmbeddedDocument,
    MetadataFilter,
    QueryResults,
    ScoredMatch,
)

if TYPE_CHECKING:
    from types import TracebackType


class ChromaVectorStore:
    """
    ChromaDB adapter implementing VectorStorePort.

    Instantiated via the factory in :mod:`jobsearch_rag.rag.ports`::

        from jobsearch_rag.rag.ports import VectorStoreConfig, create_vector_store

        store = create_vector_store(VectorStoreConfig(
            persist_dir="./data/chroma_db",
            distance_metric="cosine",
            sync_threshold=1000,
        ))
    """

    def __init__(self, persist_dir: str, distance_metric: str, sync_threshold: int) -> None:
        """Initialize ChromaDB client at *persist_dir*."""
        self.persist_dir = persist_dir
        self._distance_metric = distance_metric
        self._sync_threshold = sync_threshold
        self._client = chromadb.PersistentClient(path=persist_dir)
        logger.debug("ChromaDB client initialized at %s", persist_dir)

    def __enter__(self) -> Self:
        """Enter the runtime context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context, releasing resources."""
        self.close()

    def close(self) -> None:
        """
        Release ChromaDB file handles (SQLite, HNSW segments).

        Must be called before deleting the *persist_dir* on Windows,
        where POSIX unlink-while-open semantics are unavailable.
        """
        self._client.clear_system_cache()

    # -- Collection lifecycle ------------------------------------------------

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Return the named ChromaDB collection, creating if necessary.

        Uses the configured distance metric as the distance function.
        """
        collection = self._client.get_or_create_collection(
            name=name,
            configuration={
                "hnsw": {
                    "space": self._distance_metric,
                    "sync_threshold": self._sync_threshold,
                },
            },
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
        self._get_or_create_collection(name)
        logger.debug("Collection '%s' recreated empty", name)

    # -- Document operations -------------------------------------------------

    def add_documents(
        self,
        collection_name: str,
        *,
        documents: list[EmbeddedDocument],
    ) -> None:
        """
        Add (or update) documents with pre-computed embeddings.

        Documents with existing IDs are **upserted** (updated in place).

        Args:
            collection_name: Target collection (created if absent).
            documents: Documents with embeddings ready for storage.

        """
        ids = [doc.id for doc in documents]
        texts = [doc.document for doc in documents]
        embeddings: list[list[float]] = [doc.embedding for doc in documents]

        # ChromaDB rejects empty dicts; use None for docs without metadata
        raw_metas = [doc.metadata if doc.metadata else None for doc in documents]
        metadatas: list[dict[str, Any] | None] | None = (
            None if all(m is None for m in raw_metas) else raw_metas
        )

        collection = self._get_or_create_collection(collection_name)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
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

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)
        result = collection.get(ids=ids, include=["documents", "metadatas"])
        return [
            DocumentRecord(
                id=rid,
                document=doc or "",
                metadata=meta or {},
            )
            for rid, doc, meta in zip(
                result["ids"],
                result["documents"] or [],
                result["metadatas"] or [],
                strict=True,
            )
        ]

    def get_all_documents(
        self,
        collection_name: str,
    ) -> list[DocumentRecord]:
        """
        Retrieve all documents in the named collection.

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)
        result = collection.get(include=["documents", "metadatas"])
        return [
            DocumentRecord(
                id=rid,
                document=doc or "",
                metadata=meta or {},
            )
            for rid, doc, meta in zip(
                result["ids"],
                result["documents"] or [],
                result["metadatas"] or [],
                strict=True,
            )
        ]

    def get_by_metadata(
        self,
        collection_name: str,
        *,
        where: MetadataFilter,
    ) -> list[DocumentRecord]:
        """
        Retrieve documents matching a metadata filter.

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)
        # Convert domain filter to ChromaDB where clause
        if where.operator == "eq":
            chroma_where = {where.field: where.value}
        else:  # "ne"
            chroma_where = {where.field: {"$ne": where.value}}
        result = collection.get(where=chroma_where, include=["documents", "metadatas"])
        return [
            DocumentRecord(
                id=rid,
                document=doc or "",
                metadata=meta or {},
            )
            for rid, doc, meta in zip(
                result["ids"],
                result["documents"] or [],
                result["metadatas"] or [],
                strict=True,
            )
        ]

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
    ) -> QueryResults:
        """
        Find the *n_results* most similar documents to *query_embedding*.

        Distances are cosine distances (lower = more similar;
        0.0 = identical direction).

        Raises :class:`~jobsearch_rag.errors.ActionableError` (INDEX)
        if the collection does not exist.
        """
        collection = self._get_existing_collection(collection_name)

        # ChromaDB raises if n_results > count; clamp to available
        count = collection.count()
        if count == 0:
            return QueryResults(matches=[])

        effective_n = min(n_results, count)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_n,
            include=["documents", "metadatas", "distances"],
        )

        matches: list[ScoredMatch] = []
        ids_list: list[str] = result["ids"][0] if result["ids"] else []
        docs_list: list[str] = result["documents"][0] if result["documents"] else []
        metas_list: list[dict[str, Any]] = result["metadatas"][0] if result["metadatas"] else []
        dists_list: list[float] = result["distances"][0] if result["distances"] else []

        for rid, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list, strict=True):
            matches.append(
                ScoredMatch(
                    id=rid,
                    document=doc or "",
                    metadata=meta or {},
                    distance=dist,
                )
            )

        return QueryResults(matches=matches)

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

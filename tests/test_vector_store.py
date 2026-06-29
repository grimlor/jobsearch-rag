"""
VectorStore tests — ChromaDB collection management and querying.

Maps to BDD specs: TestCollectionLifecycle, TestDocumentOperations,
TestSimilarityQuery, TestStoreErrors, TestMetadataQuery, TestVectorStoreFactory

Spec classes:
    TestCollectionLifecycle
    TestDocumentOperations
    TestSimilarityQuery
    TestStoreErrors
    TestMetadataQuery
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.ports import (
    EmbeddedDocument,
    MetadataFilter,
    VectorStoreConfig,
    VectorStorePort,
    create_vector_store,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    "Staff Platform Architect designing distributed systems",
    "Principal Data Engineer building streaming pipelines",
    "Developer Relations evangelist creating SDK documentation",
]

SAMPLE_IDS = ["doc-1", "doc-2", "doc-3"]

SAMPLE_METADATA = [
    {"source": "resume", "section": "experience"},
    {"source": "resume", "section": "experience"},
    {"source": "resume", "section": "skills"},
]

# Fake embedding vectors — 5 dimensions is enough for tests.
# Vectors are directionally meaningful so similarity tests work:
#   doc-1 and doc-3 are somewhat similar (both about leadership)
#   doc-2 points in a different direction (data engineering)
EMBED_DIM = 5
EMBED_1 = [0.9, 0.1, 0.2, 0.0, 0.3]  # architect / leadership
EMBED_2 = [0.1, 0.8, 0.1, 0.7, 0.0]  # data engineering
EMBED_3 = [0.7, 0.2, 0.3, 0.0, 0.4]  # devrel / leadership-adjacent
SAMPLE_EMBEDDINGS = [EMBED_1, EMBED_2, EMBED_3]


class _NonConformingStore:
    """
    A class that accepts kwargs but does not implement VectorStorePort.

    Referenced by string in TestVectorStoreFactory tests via
    ``store_class="tests.test_vector_store._NonConformingStore"``.
    """

    def __init__(self, **kwargs: object) -> None:
        pass


# Ensure pyright recognises the class as accessed (string-referenced in tests)
_FACTORY_TEST_CLASSES = (_NonConformingStore,)


@pytest.fixture
def store() -> Iterator[VectorStorePort]:
    """Create a ChromaVectorStore (adapter test) backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = create_vector_store(
            VectorStoreConfig(
                store_class="jobsearch_rag.rag.store.ChromaVectorStore",
                persist_dir=tmpdir,
                distance_metric="cosine",
                sync_threshold=1,
            )
        )
        yield s
        s.close()


@pytest.fixture
def populated_store(store: VectorStorePort) -> VectorStorePort:
    """A VectorStorePort with three documents already added to 'test_collection'."""
    store.add_documents(
        collection_name="test_collection",
        documents=[
            EmbeddedDocument(id=id_, document=doc, embedding=emb, metadata=meta)
            for id_, doc, emb, meta in zip(
                SAMPLE_IDS, SAMPLE_DOCS, SAMPLE_EMBEDDINGS, SAMPLE_METADATA, strict=True
            )
        ],
    )
    return store


# ---------------------------------------------------------------------------
# TestCollectionLifecycle
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    """
    REQUIREMENT: Collections are created, retrieved, and reset reliably

    WHO: The indexer managing vector store collections
    WHAT: (1) The system reports zero documents for a freshly reset collection.
          (2) The system reports a document count of 3 after 3 documents are added to a collection.
          (3) The system drops all documents and reports a document count of zero when reset_collection is called on a populated collection.
          (4) The system performs a safe no-op without raising an exception when reset_collection is called for a nonexistent collection.
          (5) The system passes the configured distance metric to ChromaDB's
              collection HNSW configuration.
    WHY: Stale or phantom collections lead to scoring against outdated data —
         a silent correctness bug that's hard to diagnose

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore, reset_collection, collection_count
        Never: Patch ChromaDB internals
    """

    def test_new_collection_has_zero_documents(self, store: VectorStorePort) -> None:
        """
        GIVEN a freshly created collection
        WHEN collection_count is checked
        THEN the document count is zero
        """
        # Given: reset collectio
        store.reset_collection("empty")

        # Then: count is zero
        assert store.collection_count("empty") == 0, "New collection should have zero documents"

    def test_collection_count_reflects_added_documents(
        self, populated_store: VectorStorePort
    ) -> None:
        """
        GIVEN a collection with 3 documents added
        WHEN collection_count is checked
        THEN it returns 3
        """
        # Then: count matches added documents
        assert populated_store.collection_count("test_collection") == 3, (
            "Count should match number of added documents"
        )

    def test_reset_drops_all_documents(self, populated_store: VectorStorePort) -> None:
        """
        GIVEN a populated collection
        WHEN reset_collection is called
        THEN all documents are dropped and count returns to zero
        """
        # When: reset the collection
        populated_store.reset_collection("test_collection")

        # Then: count is zero
        assert populated_store.collection_count("test_collection") == 0, (
            "Reset should drop all documents"
        )

    def test_reset_nonexistent_collection_does_not_raise(self, store: VectorStorePort) -> None:
        """
        GIVEN a collection name that doesn't exist
        WHEN reset_collection is called
        THEN it is a safe no-op — no exception is raised
        """
        # When/Then: reset non-existent collection (should not raise)
        store.reset_collection("never_existed")

    def test_collection_uses_configured_distance_metric(self) -> None:
        """
        GIVEN a VectorStore initialized with distance_metric = "l2"
        WHEN documents are added and queried
        THEN the returned distances reflect L2 (Euclidean) distance, not cosine
        """
        # Given: two stores with different distance metrics, same data
        docs = [
            EmbeddedDocument(id="doc-a", document="alpha", embedding=EMBED_1, metadata={"k": "v"}),
            EmbeddedDocument(id="doc-b", document="beta", embedding=EMBED_2, metadata={"k": "v"}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir_l2:
            store_l2 = create_vector_store(
                VectorStoreConfig(
                    store_class="jobsearch_rag.rag.store.ChromaVectorStore",
                    persist_dir=tmpdir_l2,
                    distance_metric="l2",
                    sync_threshold=1,
                )
            )
            store_l2.add_documents("coll", documents=docs)
            results_l2 = store_l2.query("coll", query_embedding=EMBED_1, n_results=2)
            store_l2.close()

        with tempfile.TemporaryDirectory() as tmpdir_cos:
            store_cos = create_vector_store(
                VectorStoreConfig(
                    store_class="jobsearch_rag.rag.store.ChromaVectorStore",
                    persist_dir=tmpdir_cos,
                    distance_metric="cosine",
                    sync_threshold=1,
                )
            )
            store_cos.add_documents("coll", documents=docs)
            results_cos = store_cos.query("coll", query_embedding=EMBED_1, n_results=2)
            store_cos.close()

        # Then: distances differ between L2 and cosine for the same vectors
        l2_distances = [m.distance for m in results_l2.matches]
        cos_distances = [m.distance for m in results_cos.matches]
        assert l2_distances != cos_distances, (
            f"L2 and cosine should produce different distances, "
            f"got L2={l2_distances}, cosine={cos_distances}"
        )


# ---------------------------------------------------------------------------
# TestDocumentOperations
# ---------------------------------------------------------------------------


class TestDocumentOperations:
    """
    REQUIREMENT: Documents can be added and retrieved with metadata

    WHO: The indexer populating collections with resume chunks and archetypes
    WHAT: (1) The system returns the matching document when get_documents is called with a specific ID.
          (2) The system preserves and returns the correct metadata when a document is retrieved by ID.
          (3) The system updates an existing document instead of creating a duplicate when add_documents is called with the same ID.
          (4) The system stores and returns documents when add_documents is called with a list of EmbeddedDocument objects.
    WHY: Duplicate documents inflate similarity results; lost metadata
         prevents score explanation and debugging

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore.add_documents, get_documents, collection_count
        Never: Patch ChromaDB internals or embedding storage
    """

    def test_documents_are_retrievable_by_id(self, populated_store: VectorStorePort) -> None:
        """
        GIVEN a populated collection
        WHEN get_documents is called with a specific ID
        THEN the matching document is returned
        """
        # When: retrieve by ID
        result = populated_store.get_documents("test_collection", ids=["doc-1"])

        # Then: document is returned
        assert len(result) == 1, "Should return exactly one document"
        assert "Staff Platform Architect" in result[0].document, (
            "Returned document should match the original"
        )

    def test_metadata_is_preserved(self, populated_store: VectorStorePort) -> None:
        """
        GIVEN a document added with metadata
        WHEN retrieved by ID
        THEN the metadata is preserved and correct
        """
        # When: retrieve document
        result = populated_store.get_documents("test_collection", ids=["doc-3"])

        # Then: metadata preserved
        assert result[0].metadata["section"] == "skills", (
            "Metadata should be preserved on retrieval"
        )

    def test_add_with_duplicate_id_updates_document(
        self, populated_store: VectorStorePort
    ) -> None:
        """
        GIVEN a document already in the collection
        WHEN add_documents is called with the same ID
        THEN it updates rather than creating a duplicate
        """
        # When: add with existing ID
        populated_store.add_documents(
            collection_name="test_collection",
            documents=[
                EmbeddedDocument(
                    id="doc-1",
                    document="Updated architect description",
                    embedding=EMBED_1,
                    metadata={"source": "resume", "section": "updated"},
                ),
            ],
        )

        # Then: count unchanged, document updated
        assert populated_store.collection_count("test_collection") == 3, (
            "Duplicate ID should update, not append"
        )
        result = populated_store.get_documents("test_collection", ids=["doc-1"])
        assert "Updated" in result[0].document, "Document text should be updated"

    def test_add_documents_with_embedded_document_interface(self, store: VectorStorePort) -> None:
        """
        GIVEN a list of EmbeddedDocument object
        WHEN add_documents is called
        THEN documents are stored and retrievable
        """
        store.add_documents(
            collection_name="test_collection",
            documents=[
                EmbeddedDocument(
                    id="doc-1",
                    document="First document",
                    embedding=EMBED_1,
                    metadata={"key": "value"},
                ),
                EmbeddedDocument(
                    id="doc-2",
                    document="Second document",
                    embedding=EMBED_2,
                ),
            ],
        )

        results = store.get_documents("test_collection", ids=["doc-1", "doc-2"])
        assert len(results) == 2, f"Expected 2 documents, got {len(results)}"


# ---------------------------------------------------------------------------
# TestSimilarityQuery
# ---------------------------------------------------------------------------


class TestSimilarityQuery:
    """
    REQUIREMENT: Similarity queries return documents ranked by closeness

    WHO: The scorer computing fit, archetype, and history scores
    WHAT: (1) The system returns the architect document first when the query vector is most similar to it.
          (2) The system includes similarity distance scores in the query results as a list of floats.
          (3) The system limits the query output to one result when `n_results=1`.
          (4) The system returns an empty result for an empty collection instead of raising an error.
          (5) The system includes the original document text and metadata in the query results.
    WHY: Incorrect similarity ordering would silently invert job rankings —
         the most dangerous class of bug in the system

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore.query, similarity ranking, distance computation
        Never: Patch ChromaDB query internals or distance functions
    """

    def test_query_returns_most_similar_document_first(
        self, populated_store: VectorStorePort
    ) -> None:
        """
        GIVEN a populated collection with directional embeddings
        WHEN querying with an architect-like vector
        THEN the architect document is returned first
        """
        # When: query with architect-direction vector
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,
            n_results=3,
        )

        # Then: doc-1 (architect) is most similar
        assert results.matches[0].id == "doc-1", "Architect document should be most similar"

    def test_query_returns_similarity_distances(self, populated_store: VectorStorePort) -> None:
        """
        GIVEN a populated collection
        WHEN a similarity query is run
        THEN results include distance scores as a list of floats
        """
        # When: query
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,
            n_results=2,
        )

        # Then: distances are floats
        distances = [m.distance for m in results.matches]
        assert len(distances) == 2, "Should return 2 distance values"
        assert all(isinstance(d, float) for d in distances), "Distances should be floats"

    def test_n_results_limits_output(self, populated_store: VectorStorePort) -> None:
        """
        GIVEN a collection with 3 documents
        WHEN querying with n_results=1
        THEN only 1 result is returned
        """
        # When: query with limit
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,
            n_results=1,
        )

        # Then: exactly 1 result
        assert len(results.matches) == 1, "Should return exactly 1 result"

    def test_query_empty_collection_returns_empty(self, store: VectorStorePort) -> None:
        """
        GIVEN an empty collection
        WHEN a similarity query is run
        THEN an empty result is returned, not an error
        """
        # Given: empty collection
        store.reset_collection("empty")

        # When: query empty collection
        results = store.query(
            collection_name="empty",
            query_embedding=EMBED_1,
            n_results=5,
        )

        # Then: empty results
        assert results.matches == [], "Empty collection should return empty results"

    def test_query_includes_document_text_and_metadata(
        self, populated_store: VectorStorePort
    ) -> None:
        """
        GIVEN a populated collection
        WHEN a similarity query is run
        THEN results include the original document text and metadata
        """
        # When: query
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,
            n_results=1,
        )

        # Then: document and metadata present
        assert results.matches[0].document is not None, "Document text should be included"
        assert results.matches[0].metadata is not None, "Metadata should be included"


# ---------------------------------------------------------------------------
# TestStoreErrors
# ---------------------------------------------------------------------------


class TestStoreErrors:
    """
    REQUIREMENT: Store errors are actionable and classified correctly

    WHO: The pipeline runner catching errors to present clear guidance
    WHAT: (1) The system raises an INDEX error that tells the operator to run the index command when `query` is called on a nonexistent collection.
          (2) The system names the nonexistent collection and provides step-by-step guidance in the INDEX error when `query` is called.
          (3) The system raises an INDEX error with actionable guidance when `get_documents` is called on a nonexistent collection.
          (4) The system raises an INDEX error with actionable guidance when `collection_count` is called on a nonexistent collection.
    WHY: Generic exceptions force operators to read stack traces —
         actionable errors tell them exactly what to fix

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore error paths, ActionableError classification
        Never: Patch error construction or ErrorType enum
    """

    def test_query_nonexistent_collection_tells_operator_to_run_index(
        self, store: VectorStorePort
    ) -> None:
        """
        GIVEN a collection that doesn't exist
        WHEN query is called
        THEN an INDEX error is raised telling the operator to run the index command
        """
        # When/Then: query nonexistent collection raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            store.query(
                collection_name="nonexistent",
                query_embedding=EMBED_1,
                n_results=5,
            )

        # Then: error is INDEX with guidance
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX, "Error type should be INDEX"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"
        assert len(err.troubleshooting.steps) > 0, "Troubleshooting should have steps"

    def test_index_error_names_collection_and_provides_guidance(
        self, store: VectorStorePort
    ) -> None:
        """
        GIVEN a nonexistent collection
        WHEN query is called
        THEN the INDEX error names the collection and provides step-by-step guidance
        """
        # When/Then: query nonexistent collection
        with pytest.raises(ActionableError) as exc_info:
            store.query(
                collection_name="nonexistent",
                query_embedding=EMBED_1,
                n_results=5,
            )

        # Then: error names the collection
        err = exc_info.value
        assert "nonexistent" in err.error, "Error should name the collection"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_get_documents_nonexistent_collection_provides_guidance(
        self, store: VectorStorePort
    ) -> None:
        """
        GIVEN a nonexistent collection
        WHEN get_documents is called
        THEN an INDEX error is raised with actionable guidance
        """
        # When/Then: get from nonexistent collection
        with pytest.raises(ActionableError) as exc_info:
            store.get_documents("nonexistent", ids=["doc-1"])

        # Then: error is INDEX with guidance
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX, "Error type should be INDEX"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_collection_count_nonexistent_provides_guidance(self, store: VectorStorePort) -> None:
        """
        GIVEN a nonexistent collection
        WHEN collection_count is called
        THEN an INDEX error is raised with actionable guidance
        """
        # When/Then: count nonexistent collection
        with pytest.raises(ActionableError) as exc_info:
            store.collection_count("nonexistent")

        # Then: error is INDEX with guidance
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX, "Error type should be INDEX"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"


# ---------------------------------------------------------------------------
# TestMetadataQuery
# ---------------------------------------------------------------------------


class TestMetadataQuery:
    """
    REQUIREMENT: Documents can be retrieved by metadata filter

    WHO: The scorer retrieving past rejection reasons for disqualifier augmentation
    WHAT: (1) The system returns only documents whose metadata matches the requested value.
          (2) The system returns an empty result when no documents match the requested metadata value.
          (3) The system raises an actionable INDEX error when get_by_metadata is called on a nonexistent collection.
    WHY: The disqualifier prompt needs past 'no' reasons to learn the operator's
         personal rejection patterns — metadata queries make this possible

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore.get_by_metadata, add_documents, metadata filtering
        Never: Patch ChromaDB metadata internals
    """

    def test_get_by_metadata_returns_matching_documents(
        self, populated_store: VectorStorePort
    ) -> None:
        """
        GIVEN a collection with mixed metadata values
        WHEN get_by_metadata filters on a specific value
        THEN only matching documents are returned
        """
        # Given: documents with different verdict metadata
        populated_store.add_documents(
            collection_name="decisions",
            documents=[
                EmbeddedDocument(
                    id="decision-1",
                    document="Great role",
                    embedding=EMBED_1,
                    metadata={"verdict": "yes", "reason": ""},
                ),
                EmbeddedDocument(
                    id="decision-2",
                    document="Bad role",
                    embedding=EMBED_2,
                    metadata={"verdict": "no", "reason": "on-call required"},
                ),
                EmbeddedDocument(
                    id="decision-3",
                    document="Another bad role",
                    embedding=EMBED_3,
                    metadata={"verdict": "no", "reason": "fully on-site"},
                ),
            ],
        )

        # When: filter by verdict=no
        results = populated_store.get_by_metadata(
            "decisions",
            where=MetadataFilter("verdict", "eq", "no"),
        )

        # Then: only 'no' verdicts returned
        assert len(results) == 2, "Should return 2 matching documents"
        reasons = [doc.metadata["reason"] for doc in results]
        assert "on-call required" in reasons, "First reason should match"
        assert "fully on-site" in reasons, "Second reason should match"

    def test_get_by_metadata_returns_empty_when_no_match(
        self, populated_store: VectorStorePort
    ) -> None:
        """
        GIVEN a collection with documents
        WHEN get_by_metadata filters on a value that matches nothing
        THEN an empty result is returned
        """
        # Given: a collection with only 'yes' verdicts
        populated_store.add_documents(
            collection_name="decisions",
            documents=[
                EmbeddedDocument(
                    id="decision-only-yes",
                    document="A good role",
                    embedding=EMBED_1,
                    metadata={"verdict": "yes", "reason": ""},
                ),
            ],
        )

        # When: filter by verdict=no
        results = populated_store.get_by_metadata(
            "decisions",
            where=MetadataFilter("verdict", "eq", "no"),
        )

        # Then: no documents match
        assert len(results) == 0, "Should return no matching documents"

    def test_get_by_metadata_nonexistent_collection_raises_index_error(
        self, store: VectorStorePort
    ) -> None:
        """
        GIVEN a nonexistent collection
        WHEN get_by_metadata is called
        THEN an actionable INDEX error is raised
        """
        # When/Then: metadata query on nonexistent collection
        with pytest.raises(ActionableError) as exc_info:
            store.get_by_metadata(
                "nonexistent_collection",
                where=MetadataFilter("verdict", "eq", "no"),
            )

        # Then: error is INDEX with suggestion
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX, "Error type should be INDEX"
        assert err.suggestion is not None, "Should include a suggestion"


# ---------------------------------------------------------------------------
# TestContextManager
# ---------------------------------------------------------------------------


class TestContextManager:
    """
    REQUIREMENT: ChromaVectorStore supports context manager protocol

    WHO: Any caller that uses `with create_vector_store(...)` syntax
    WHAT: (1) The system returns the store instance when entering the context.
          (2) The system closes gracefully when exiting the context.
    WHY: Resource leaks on Windows if file handles are not released

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore.__enter__, __exit__, close
        Never: Patch ChromaDB internals
    """

    def test_context_manager_returns_self_on_enter(self) -> None:
        """
        GIVEN a ChromaVectorStore
        WHEN used as a context manager
        THEN __enter__ returns the store itself
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_vector_store(
                VectorStoreConfig(
                    store_class="jobsearch_rag.rag.store.ChromaVectorStore",
                    persist_dir=tmpdir,
                    distance_metric="cosine",
                    sync_threshold=1,
                )
            )
            with store as ctx:
                assert ctx is store, "Context manager should return self"

    def test_context_manager_closes_on_exit(self) -> None:
        """
        GIVEN a ChromaVectorStore used as a context manager
        WHEN the context exits
        THEN the store is closed without error
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_vector_store(
                VectorStoreConfig(
                    store_class="jobsearch_rag.rag.store.ChromaVectorStore",
                    persist_dir=tmpdir,
                    distance_metric="cosine",
                    sync_threshold=1,
                )
            )
            with store:
                store.reset_collection("ctx_test")
                assert store.collection_count("ctx_test") == 0


# ---------------------------------------------------------------------------
# TestGetAll
# ---------------------------------------------------------------------------


class TestGetAll:
    """
    REQUIREMENT: All documents in a collection can be retrieved

    WHO: The CLI exporting full collection contents
    WHAT: (1) The system returns all documents with their metadata.
          (2) The system raises an actionable INDEX error for nonexistent collections.
    WHY: Bulk retrieval is required for export and audit operations

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore.get_all, add_documents
        Never: Patch ChromaDB internals
    """

    def test_get_all_returns_all_documents(self, populated_store: VectorStorePort) -> None:
        """
        GIVEN a collection with 3 documents
        WHEN get_all is called
        THEN all 3 documents are returned with correct metadata
        """
        results = populated_store.get_all_documents("test_collection")

        assert len(results) == 3, "Should return all 3 documents"
        returned_ids = {r.id for r in results}
        assert returned_ids == {"doc-1", "doc-2", "doc-3"}, "All IDs should be present"

    def test_get_all_nonexistent_collection_raises_index_error(
        self, store: VectorStorePort
    ) -> None:
        """
        GIVEN a nonexistent collection
        WHEN get_all is called
        THEN an actionable INDEX error is raised
        """
        with pytest.raises(ActionableError) as exc_info:
            store.get_all_documents("nonexistent_collection")

        err = exc_info.value
        assert err.error_type == ErrorType.INDEX, "Error type should be INDEX"


# ---------------------------------------------------------------------------
# TestDeleteById
# ---------------------------------------------------------------------------


class TestDeleteById:
    """
    REQUIREMENT: Documents can be deleted by ID

    WHO: The rescore pipeline removing stale entries
    WHAT: (1) The system removes the specified documents from the collection.
          (2) The system raises an actionable INDEX error for nonexistent collections.
    WHY: Stale entries must be purged before re-indexing to avoid ghost scores

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore.delete_by_id, collection_count
        Never: Patch ChromaDB internals
    """

    def test_delete_by_id_removes_documents(self, populated_store: VectorStorePort) -> None:
        """
        GIVEN a collection with 3 documents
        WHEN delete_by_id is called with 2 IDs
        THEN only 1 document remains
        """
        populated_store.delete_by_id("test_collection", ids=["doc-1", "doc-2"])

        assert populated_store.collection_count("test_collection") == 1, (
            "Should have 1 document remaining"
        )

    def test_delete_by_id_nonexistent_collection_raises_index_error(
        self, store: VectorStorePort
    ) -> None:
        """
        GIVEN a nonexistent collection
        WHEN delete_by_id is called
        THEN an actionable INDEX error is raised
        """
        with pytest.raises(ActionableError) as exc_info:
            store.delete_by_id("nonexistent_collection", ids=["doc-1"])

        err = exc_info.value
        assert err.error_type == ErrorType.INDEX, "Error type should be INDEX"


# ---------------------------------------------------------------------------
# TestMetadataQueryNe
# ---------------------------------------------------------------------------


class TestMetadataQueryNe:
    """
    REQUIREMENT: Documents can be filtered by metadata not-equal operator

    WHO: The decisions module filtering out empty reason fields
    WHAT: (1) The system returns only documents whose metadata does NOT match the filter value.
    WHY: The audit_decisions function requires "ne" filtering to exclude empty reasons

    MOCK BOUNDARY:
        Mock: nothing — uses real ChromaDB via tmpdir (adapter test)
        Real: ChromaVectorStore.get_by_metadata with ne operator
        Never: Patch ChromaDB internals
    """

    def test_get_by_metadata_ne_returns_non_matching_documents(
        self, store: VectorStorePort
    ) -> None:
        """
        GIVEN a collection with documents having different metadata values
        WHEN get_by_metadata filters with operator "ne"
        THEN only documents NOT matching the value are returned
        """
        # Given: documents with mixed verdict values
        store.add_documents(
            collection_name="filter_test",
            documents=[
                EmbeddedDocument(
                    id="d-1",
                    document="Yes role",
                    embedding=EMBED_1,
                    metadata={"verdict": "yes", "reason": "good fit"},
                ),
                EmbeddedDocument(
                    id="d-2",
                    document="No role",
                    embedding=EMBED_2,
                    metadata={"verdict": "no", "reason": ""},
                ),
                EmbeddedDocument(
                    id="d-3",
                    document="Another yes",
                    embedding=EMBED_3,
                    metadata={"verdict": "yes", "reason": "great team"},
                ),
            ],
        )

        # When: filter by reason != ""
        results = store.get_by_metadata(
            "filter_test",
            where=MetadataFilter(field="reason", operator="ne", value=""),
        )

        # Then: only documents with non-empty reason returned
        assert len(results) == 2, "Should return 2 documents with non-empty reason"
        returned_ids = {r.id for r in results}
        assert returned_ids == {"d-1", "d-3"}, "Should include d-1 and d-3"


# ---------------------------------------------------------------------------
# TestVectorStoreFactory
# ---------------------------------------------------------------------------


class TestVectorStoreFactory:
    """
    REQUIREMENT: The reflection-based factory rejects non-conforming classes at startup

    WHO: The operator who misconfigures store_class in settings.toml
    WHAT: (1) The system raises a TypeError naming the non-conforming class when create_vector_store is called with a store_class that does not implement VectorStorePort.
          (2) The system successfully instantiates and returns a VectorStorePort when create_vector_store is called with a conforming store_class.
    WHY: A non-conforming store_class would fail deep in a pipeline run with
         an opaque AttributeError — catching at startup turns a debugging session
         into a one-line config fix

    MOCK BOUNDARY:
        Mock: nothing
        Real: create_vector_store, VectorStorePort runtime check
        Never: Patch importlib or isinstance
    """

    def test_factory_rejects_non_conforming_class(self) -> None:
        """
        GIVEN a VectorStoreConfig with store_class pointing at a non-conforming class
        WHEN create_vector_store is called
        THEN TypeError is raised with the class name in the message
        """
        # Given: config pointing at a class that accepts kwargs but doesn't implement VectorStorePort
        config = VectorStoreConfig(
            persist_dir="/tmp/unused",
            distance_metric="cosine",
            sync_threshold=1,
            store_class="tests.test_vector_store._NonConformingStore",
        )

        # When/Then: factory raises TypeError
        with pytest.raises(TypeError) as exc_info:
            create_vector_store(config)

        # Then: error names the class
        assert "_NonConformingStore" in str(exc_info.value), (
            "TypeError should name the non-conforming class"
        )

    def test_factory_returns_port_for_conforming_class(self) -> None:
        """
        GIVEN a VectorStoreConfig with store_class pointing at FakeVectorStore
        WHEN create_vector_store is called
        THEN a VectorStorePort instance is returned
        """
        # Given: config pointing at FakeVectorStore
        config = VectorStoreConfig(
            persist_dir="/tmp/unused",
            distance_metric="cosine",
            sync_threshold=1,
            store_class="tests.fakes.FakeVectorStore",
        )

        # When: factory creates the store
        store = create_vector_store(config)

        # Then: returns a VectorStorePort instance
        assert isinstance(store, VectorStorePort), (
            f"Expected VectorStorePort instance, got {type(store)}"
        )

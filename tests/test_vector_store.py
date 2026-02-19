"""VectorStore tests — ChromaDB collection management and querying.

Maps to BDD specs: TestCollectionLifecycle, TestDocumentOperations,
TestSimilarityQuery, TestStoreErrors
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.store import VectorStore

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


@pytest.fixture
def store() -> Iterator[VectorStore]:
    """Create a VectorStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


@pytest.fixture
def populated_store(store: VectorStore) -> VectorStore:
    """A VectorStore with three documents already added to 'test_collection'."""
    store.add_documents(
        collection_name="test_collection",
        ids=SAMPLE_IDS,
        documents=SAMPLE_DOCS,
        embeddings=SAMPLE_EMBEDDINGS,
        metadatas=SAMPLE_METADATA,
    )
    return store


# ---------------------------------------------------------------------------
# TestCollectionLifecycle
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    """REQUIREMENT: Collections are created, retrieved, and reset reliably.

    WHO: The indexer managing ChromaDB collections
    WHAT: get_or_create returns a usable collection; calling it twice returns
          the same collection; reset drops all documents; collection_count
          reflects the current state accurately
    WHY: Stale or phantom collections lead to scoring against outdated data —
         a silent correctness bug that's hard to diagnose
    """

    def test_get_or_create_returns_collection(self, store: VectorStore) -> None:
        """A new collection is created on first call to get_or_create_collection."""
        collection = store.get_or_create_collection("new_collection")
        assert collection is not None

    def test_get_or_create_is_idempotent(self, store: VectorStore) -> None:
        """Calling get_or_create_collection twice returns the same collection, not a duplicate."""
        c1 = store.get_or_create_collection("same_name")
        c2 = store.get_or_create_collection("same_name")
        assert c1.name == c2.name

    def test_new_collection_has_zero_documents(self, store: VectorStore) -> None:
        """A freshly created collection has a document count of zero."""
        store.get_or_create_collection("empty")
        assert store.collection_count("empty") == 0

    def test_collection_count_reflects_added_documents(
        self, populated_store: VectorStore
    ) -> None:
        """Document count matches the number of documents added."""
        assert populated_store.collection_count("test_collection") == 3

    def test_reset_drops_all_documents(self, populated_store: VectorStore) -> None:
        """Resetting a collection drops all its documents, returning count to zero."""
        populated_store.reset_collection("test_collection")
        assert populated_store.collection_count("test_collection") == 0

    def test_reset_nonexistent_collection_does_not_raise(
        self, store: VectorStore
    ) -> None:
        """Resetting a collection that doesn't exist is a safe no-op."""
        store.reset_collection("never_existed")  # Should not raise


# ---------------------------------------------------------------------------
# TestDocumentOperations
# ---------------------------------------------------------------------------


class TestDocumentOperations:
    """REQUIREMENT: Documents can be added and retrieved with metadata.

    WHO: The indexer populating collections with resume chunks and archetypes
    WHAT: Documents are stored with their IDs, embeddings, and metadata;
          adding duplicate IDs updates rather than duplicates; metadata
          is preserved and queryable
    WHY: Duplicate documents inflate similarity results; lost metadata
         prevents score explanation and debugging
    """

    def test_documents_are_retrievable_by_id(
        self, populated_store: VectorStore
    ) -> None:
        """Documents added to a collection can be retrieved by their IDs."""
        result = populated_store.get_documents("test_collection", ids=["doc-1"])
        assert len(result["documents"]) == 1
        assert "Staff Platform Architect" in result["documents"][0]

    def test_metadata_is_preserved(self, populated_store: VectorStore) -> None:
        """Metadata attached to a document is preserved and retrievable."""
        result = populated_store.get_documents("test_collection", ids=["doc-3"])
        assert result["metadatas"][0]["section"] == "skills"

    def test_add_with_duplicate_id_updates_document(
        self, populated_store: VectorStore
    ) -> None:
        """Adding a document with an existing ID updates it rather than creating a duplicate."""
        populated_store.add_documents(
            collection_name="test_collection",
            ids=["doc-1"],
            documents=["Updated architect description"],
            embeddings=[EMBED_1],
            metadatas=[{"source": "resume", "section": "updated"}],
        )
        assert populated_store.collection_count("test_collection") == 3
        result = populated_store.get_documents("test_collection", ids=["doc-1"])
        assert "Updated" in result["documents"][0]

    def test_add_documents_with_mismatched_lengths_names_the_mismatch(
        self, store: VectorStore
    ) -> None:
        """Mismatched lengths produce a VALIDATION error naming the mismatch so the caller can fix it."""
        with pytest.raises(ActionableError) as exc_info:
            store.add_documents(
                collection_name="test_collection",
                ids=["doc-1"],
                documents=["one", "two"],  # 2 docs but only 1 id
                embeddings=[EMBED_1, EMBED_2],
            )
        err = exc_info.value
        assert err.error_type == ErrorType.VALIDATION
        assert err.suggestion is not None
        assert err.troubleshooting is not None


# ---------------------------------------------------------------------------
# TestSimilarityQuery
# ---------------------------------------------------------------------------


class TestSimilarityQuery:
    """REQUIREMENT: Similarity queries return documents ranked by closeness.

    WHO: The scorer computing fit, archetype, and history scores
    WHAT: Querying with a vector returns documents ordered by similarity;
          the most similar document appears first; similarity scores are
          floats; the n_results parameter limits output; querying an
          empty collection returns an empty result
    WHY: Incorrect similarity ordering would silently invert job rankings —
         the most dangerous class of bug in the system
    """

    def test_query_returns_most_similar_document_first(
        self, populated_store: VectorStore
    ) -> None:
        """Querying with architect-like vector returns the architect document first."""
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,  # architect direction
            n_results=3,
        )
        # doc-1 (architect) should be most similar to EMBED_1
        assert results["ids"][0][0] == "doc-1"

    def test_query_returns_similarity_distances(
        self, populated_store: VectorStore
    ) -> None:
        """Query results include distance scores as a list of floats."""
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,
            n_results=2,
        )
        distances = results["distances"][0]
        assert len(distances) == 2
        assert all(isinstance(d, float) for d in distances)

    def test_n_results_limits_output(
        self, populated_store: VectorStore
    ) -> None:
        """Requesting fewer results than available limits the output correctly."""
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,
            n_results=1,
        )
        assert len(results["ids"][0]) == 1

    def test_query_empty_collection_returns_empty(
        self, store: VectorStore
    ) -> None:
        """Querying an empty collection returns an empty result, not an error."""
        store.get_or_create_collection("empty")
        results = store.query(
            collection_name="empty",
            query_embedding=EMBED_1,
            n_results=5,
        )
        assert results["ids"][0] == []

    def test_query_includes_document_text_and_metadata(
        self, populated_store: VectorStore
    ) -> None:
        """Query results include the original document text and metadata."""
        results = populated_store.query(
            collection_name="test_collection",
            query_embedding=EMBED_1,
            n_results=1,
        )
        assert results["documents"][0][0] is not None
        assert results["metadatas"][0][0] is not None


# ---------------------------------------------------------------------------
# TestStoreErrors
# ---------------------------------------------------------------------------


class TestStoreErrors:
    """REQUIREMENT: Store errors are actionable and classified correctly.

    WHO: The pipeline runner catching errors to present clear guidance
    WHAT: Querying a nonexistent collection raises an INDEX error;
          the error message names the collection; invalid persist_dir
          raises a CONFIG error
    WHY: Generic exceptions force operators to read stack traces —
         actionable errors tell them exactly what to fix
    """

    def test_query_nonexistent_collection_tells_operator_to_run_index(
        self, store: VectorStore
    ) -> None:
        """Querying a nonexistent collection tells the operator to run the index command."""
        with pytest.raises(ActionableError) as exc_info:
            store.query(
                collection_name="nonexistent",
                query_embedding=EMBED_1,
                n_results=5,
            )
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    def test_index_error_names_collection_and_provides_guidance(
        self, store: VectorStore
    ) -> None:
        """The INDEX error names the collection and provides step-by-step guidance."""
        with pytest.raises(ActionableError) as exc_info:
            store.query(
                collection_name="nonexistent",
                query_embedding=EMBED_1,
                n_results=5,
            )
        err = exc_info.value
        assert "nonexistent" in err.error
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    def test_get_documents_nonexistent_collection_provides_guidance(
        self, store: VectorStore
    ) -> None:
        """Getting documents from a nonexistent collection provides actionable guidance."""
        with pytest.raises(ActionableError) as exc_info:
            store.get_documents("nonexistent", ids=["doc-1"])
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    def test_collection_count_nonexistent_provides_guidance(
        self, store: VectorStore
    ) -> None:
        """Checking count of a nonexistent collection provides actionable guidance."""
        with pytest.raises(ActionableError) as exc_info:
            store.collection_count("nonexistent")
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX
        assert err.suggestion is not None
        assert err.troubleshooting is not None


# ---------------------------------------------------------------------------
# TestMetadataQuery
# ---------------------------------------------------------------------------


class TestMetadataQuery:
    """REQUIREMENT: Documents can be retrieved by metadata filter.

    WHO: The scorer retrieving past rejection reasons for disqualifier augmentation
    WHAT: get_by_metadata returns only documents matching the where clause;
          nonexistent collections raise an actionable INDEX error
    WHY: The disqualifier prompt needs past 'no' reasons to learn the operator's
         personal rejection patterns — metadata queries make this possible
    """

    def test_get_by_metadata_returns_matching_documents(
        self, populated_store: VectorStore
    ) -> None:
        """GIVEN a collection with mixed metadata values
        WHEN get_by_metadata filters on a specific value
        THEN only matching documents are returned.
        """
        # Add documents with different verdict metadata
        populated_store.add_documents(
            collection_name="decisions",
            ids=["decision-1", "decision-2", "decision-3"],
            documents=["Great role", "Bad role", "Another bad role"],
            embeddings=[EMBED_1, EMBED_2, EMBED_3],
            metadatas=[
                {"verdict": "yes", "reason": ""},
                {"verdict": "no", "reason": "on-call required"},
                {"verdict": "no", "reason": "fully on-site"},
            ],
        )
        results = populated_store.get_by_metadata(
            "decisions",
            where={"verdict": "no"},
            include=["metadatas"],
        )
        assert len(results["ids"]) == 2
        reasons = [m["reason"] for m in results["metadatas"]]
        assert "on-call required" in reasons
        assert "fully on-site" in reasons

    def test_get_by_metadata_returns_empty_when_no_match(
        self, populated_store: VectorStore
    ) -> None:
        """GIVEN a collection with documents
        WHEN get_by_metadata filters on a value that matches nothing
        THEN an empty result is returned.
        """
        populated_store.add_documents(
            collection_name="decisions",
            ids=["decision-only-yes"],
            documents=["A good role"],
            embeddings=[EMBED_1],
            metadatas=[{"verdict": "yes", "reason": ""}],
        )
        results = populated_store.get_by_metadata(
            "decisions",
            where={"verdict": "no"},
            include=["metadatas"],
        )
        assert len(results["ids"]) == 0

    def test_get_by_metadata_nonexistent_collection_raises_index_error(
        self, store: VectorStore
    ) -> None:
        """Querying metadata on a nonexistent collection raises an actionable INDEX error."""
        with pytest.raises(ActionableError) as exc_info:
            store.get_by_metadata(
                "nonexistent_collection",
                where={"verdict": "no"},
            )
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX
        assert err.suggestion is not None

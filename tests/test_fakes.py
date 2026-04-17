"""
BDD specs for test fakes that satisfy port protocols.

Covers: FakeEmbedder (D2), InMemoryVectorStore (D4).

Public API surface (from tests/fakes):
    FakeEmbedder(
        embed_vector: list[float] = [0.0, ...],
        classify_response: str = "{}",
        embed_side_effect: Callable | None = None,
        classify_side_effect: Callable | None = None,
    )
    await fake.embed(text: str) -> list[float]
    await fake.classify(prompt: str) -> str
    fake.embed_calls: list[str]      # recorded arguments
    fake.embed_call_count: int
    fake.classify_call_count: int

    InMemoryVectorStore()
    store.add_documents(collection_name, *, ids, documents, embeddings, metadatas=None)
    store.query(collection_name, *, query_embedding, n_results=5) -> QueryResult
    store.get_documents(collection_name, *, ids) -> GetResult
    store.get_by_metadata(collection_name, *, where, include=None) -> GetResult
    store.get_all_documents(collection_name, *, include=None) -> GetResult
    store.delete_by_id(collection_name, *, ids)
    store.collection_count(name) -> int
    store.reset_collection(name)
    store.close()

Public API surface (from src/jobsearch_rag/ports):
    EmbeddingPort  -- Protocol (embed, classify)
    HealthCheckable -- Protocol (health_check)
    MetricsProvider -- Protocol (metrics property)
    VectorStorePort -- Protocol (add_documents, query, get_documents, ...)
    QueryResult -- dataclass (ids, documents, metadatas, distances)
    GetResult -- dataclass (ids, documents, metadatas)

Public API surface (from src/jobsearch_rag/errors):
    ActionableError.index(collection_name) -- for missing collection errors
"""

from __future__ import annotations

from typing import cast

import pytest

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.ports import (
    EmbeddingPort,
    GetResult,
    HealthCheckable,
    MetricsProvider,
    QueryResult,
    VectorStorePort,
)
from tests.fakes import FakeEmbedder, InMemoryVectorStore


class TestFakeEmbedder:
    """
    REQUIREMENT: FakeEmbedder is a test double that satisfies EmbeddingPort,
    HealthCheckable (no-op), and MetricsProvider (via @observable) with
    configurable, deterministic behavior and no Ollama dependency.

    WHO: All unit tests that need an embedder -- replaces the mock_embedder
         fixture backed by patched ollama_sdk.AsyncClient.
    WHAT: (1) FakeEmbedder satisfies EmbeddingPort (isinstance check).
          (2) FakeEmbedder satisfies HealthCheckable (no-op health_check).
          (3) FakeEmbedder satisfies MetricsProvider (via @observable).
          (4) embed() returns a configurable fixed vector.
          (5) classify() returns a configurable fixed response string.
          (6) embed() can be configured with a side_effect callable for
              per-call behavior (e.g., error injection, varying vectors).
          (7) classify() can be configured with a side_effect callable.
          (8) embed() records call arguments for inspection.
          (9) embed() tracks call count.
          (10) classify() tracks call count.
    WHY: FakeEmbedder eliminates all ollama_sdk.AsyncClient patching and
         ~15 type: ignore[union-attr] suppressions from _client access.
         Tests express intent ("embedder returns this vector") instead of
         SDK internals ("mock_client.embed.return_value.embeddings = ...").
         Satisfying all three protocols lets PipelineRunner call
         health_check and collect metrics unconditionally.

    MOCK BOUNDARY:
        Mock:  Nothing -- FakeEmbedder IS the test double
        Real:  FakeEmbedder, EmbeddingPort protocol check
        Never: Mock FakeEmbedder internals
    """

    def test_fake_embedder_satisfies_embedding_port(self) -> None:
        """
        Given a FakeEmbedder instance
        When isinstance(fake, EmbeddingPort) is checked
        Then it returns True
        """
        # Given: a FakeEmbedder instance

        fake = cast("object", FakeEmbedder())

        # When: isinstance check against EmbeddingPort
        result = isinstance(fake, EmbeddingPort)

        # Then: FakeEmbedder satisfies EmbeddingPort
        assert result is True, (
            f"FakeEmbedder should satisfy EmbeddingPort. isinstance returned {result}"
        )

    def test_fake_embedder_satisfies_health_checkable(self) -> None:
        """
        Given a FakeEmbedder instance
        When isinstance(fake, HealthCheckable) is checked
        Then it returns True
        """
        # Given: a FakeEmbedder instance

        fake = cast("object", FakeEmbedder())

        # When: isinstance check against HealthCheckable
        result = isinstance(fake, HealthCheckable)

        # Then: FakeEmbedder satisfies HealthCheckable (no-op health_check)
        assert result is True, (
            f"FakeEmbedder should satisfy HealthCheckable. isinstance returned {result}"
        )

    def test_fake_embedder_satisfies_metrics_provider(self) -> None:
        """
        Given a FakeEmbedder instance (decorated with @observable)
        When isinstance(fake, MetricsProvider) is checked
        Then it returns True
        """
        # Given: a FakeEmbedder instance

        fake = cast("object", FakeEmbedder())

        # When: isinstance check against MetricsProvider
        result = isinstance(fake, MetricsProvider)

        # Then: FakeEmbedder satisfies MetricsProvider (via @observable)
        assert result is True, (
            f"FakeEmbedder should satisfy MetricsProvider (via @observable). "
            f"isinstance returned {result}"
        )

    async def test_embed_returns_configured_vector(self) -> None:
        """
        Given a FakeEmbedder configured with embed_vector=[0.1, 0.2, 0.3]
        When embed("any text") is awaited
        Then it returns [0.1, 0.2, 0.3]
        """
        # Given: a FakeEmbedder with a specific embed_vector
        fake = FakeEmbedder(embed_vector=[0.1, 0.2, 0.3])

        # When: embed is awaited
        result = await fake.embed("any text")

        # Then: the configured vector is returned
        assert result == [0.1, 0.2, 0.3], f"Expected [0.1, 0.2, 0.3], got {result}"

    async def test_classify_returns_configured_response(self) -> None:
        """
        Given a FakeEmbedder configured with classify_response='{"disqualified": false}'
        When classify("any prompt") is awaited
        Then it returns '{"disqualified": false}'
        """
        # Given: a FakeEmbedder with a specific classify_response
        fake = FakeEmbedder(classify_response='{"disqualified": false}')

        # When: classify is awaited
        result = await fake.classify("any prompt")

        # Then: the configured response is returned
        assert result == '{"disqualified": false}', (
            f"Expected '{{\"disqualified\": false}}', got {result!r}"
        )

    async def test_embed_side_effect_overrides_fixed_vector(self) -> None:
        """
        Given a FakeEmbedder with a side_effect callable on embed
        When embed() is awaited
        Then the side_effect return value is used instead of the fixed vector
        """
        # Given: a FakeEmbedder with embed_side_effect returning a custom vector
        custom_vector = [9.0, 8.0, 7.0]
        fake = FakeEmbedder(
            embed_vector=[0.0, 0.0, 0.0],
            embed_side_effect=lambda text: custom_vector,
        )

        # When: embed is awaited
        result = await fake.embed("ignored input")

        # Then: side_effect return value overrides the fixed vector
        assert result == custom_vector, (
            f"Expected side_effect vector {custom_vector}, got {result}"
        )

    async def test_classify_side_effect_overrides_fixed_response(self) -> None:
        """
        Given a FakeEmbedder with a side_effect callable on classify
        When classify() is awaited
        Then the side_effect return value is used instead of the fixed response
        """
        # Given: a FakeEmbedder with classify_side_effect returning a custom response
        custom_response = "side_effect_response"
        fake = FakeEmbedder(
            classify_response="default_response",
            classify_side_effect=lambda prompt: custom_response,
        )

        # When: classify is awaited
        result = await fake.classify("ignored prompt")

        # Then: side_effect return value overrides the fixed response
        assert result == custom_response, (
            f"Expected side_effect response {custom_response!r}, got {result!r}"
        )

    async def test_embed_records_call_arguments(self) -> None:
        """
        Given a FakeEmbedder
        When embed("specific text") is awaited
        Then the call arguments are recorded and inspectable
        """
        # Given: a FakeEmbedder
        fake = FakeEmbedder()

        # When: embed is called with specific text
        await fake.embed("specific text")

        # Then: the call argument is recorded
        assert fake.embed_calls == ["specific text"], (
            f"Expected embed_calls=['specific text'], got {fake.embed_calls}"
        )

    async def test_embed_tracks_call_count(self) -> None:
        """
        Given a FakeEmbedder
        When embed() is called 3 times
        Then the call count is 3
        """
        # Given: a FakeEmbedder
        fake = FakeEmbedder()

        # When: embed is called 3 times
        await fake.embed("one")
        await fake.embed("two")
        await fake.embed("three")

        # Then: call count is 3
        assert fake.embed_call_count == 3, (
            f"Expected embed_call_count=3, got {fake.embed_call_count}"
        )

    async def test_classify_tracks_call_count(self) -> None:
        """
        Given a FakeEmbedder
        When classify() is called 2 times
        Then the call count is 2
        """
        # Given: a FakeEmbedder
        fake = FakeEmbedder()

        # When: classify is called 2 times
        await fake.classify("prompt one")
        await fake.classify("prompt two")

        # Then: call count is 2
        assert fake.classify_call_count == 2, (
            f"Expected classify_call_count=2, got {fake.classify_call_count}"
        )


# ============================================================================
# TestInMemoryVectorStore
# ============================================================================

EMBED_DIM = 5
VEC_A = [0.9, 0.1, 0.2, 0.0, 0.3]
VEC_B = [0.1, 0.8, 0.1, 0.7, 0.0]
VEC_C = [0.7, 0.2, 0.3, 0.0, 0.4]


class TestInMemoryVectorStore:
    """
    REQUIREMENT: InMemoryVectorStore is a test double that satisfies
    VectorStorePort with dict-backed in-memory storage and cosine
    similarity search. No ChromaDB dependency.

    WHO: All unit tests that need a vector store -- replaces real ChromaDB
         temp-dir backed VectorStore for unit tests (integration tests
         retain real ChromaDB).
    WHAT: (1) InMemoryVectorStore satisfies VectorStorePort (isinstance check).
          (2) add_documents() stores documents retrievable by get_documents().
          (3) query() returns documents ranked by cosine similarity to the
              query embedding, with correct distances.
          (4) query() respects n_results to limit returned documents.
          (5) get_by_metadata() filters documents by metadata where clause.
          (6) get_all_documents() returns all documents in a collection.
          (7) delete_by_id() removes documents.
          (8) collection_count() returns the document count.
          (9) reset_collection() drops all documents in a collection.
          (10) close() is a no-op (satisfies the protocol).
          (11) Operations on non-existent collections raise ActionableError
               (same contract as VectorStore).
          (12) add_documents() with existing IDs performs upsert.
          (13) No WAL isolation issues -- writes are immediately visible
               from all references (eliminates §1e workaround).
    WHY: InMemoryVectorStore eliminates ChromaDB from unit tests. No temp
         dirs, no WAL isolation, no SDK mocking. Tests run faster and
         express vector-store interactions at the domain level.

    MOCK BOUNDARY:
        Mock:  Nothing -- InMemoryVectorStore IS the test double
        Real:  InMemoryVectorStore, VectorStorePort protocol check,
               QueryResult, GetResult
        Never: Mock InMemoryVectorStore internals
    """

    def test_in_memory_store_satisfies_vector_store_port(self) -> None:
        """
        Given an InMemoryVectorStore instance
        When isinstance(store, VectorStorePort) is checked
        Then it returns True
        """
        # Given: an InMemoryVectorStore
        store = InMemoryVectorStore()

        # When: checking protocol conformance
        result = isinstance(cast("object", store), VectorStorePort)

        # Then: it satisfies VectorStorePort
        assert result, "InMemoryVectorStore should satisfy VectorStorePort"

    def test_add_and_get_documents(self) -> None:
        """
        Given an InMemoryVectorStore
        When add_documents() is called then get_documents() with the same IDs
        Then the returned GetResult contains the original documents and metadata
        """
        # Given: an empty store
        store = InMemoryVectorStore()

        # When: add documents then retrieve
        store.add_documents(
            "test",
            ids=["d1", "d2"],
            documents=["Alpha", "Beta"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
        )
        result = store.get_documents("test", ids=["d1", "d2"])

        # Then: returned GetResult contains original data
        assert isinstance(result, GetResult), (
            f"get_documents() should return GetResult, got {type(result).__name__}"
        )
        assert result.ids == ["d1", "d2"], f"Expected ids ['d1', 'd2'], got {result.ids}"
        assert result.documents == ["Alpha", "Beta"], (
            f"Expected documents ['Alpha', 'Beta'], got {result.documents}"
        )
        assert result.metadatas == [{"k": "v1"}, {"k": "v2"}], (
            f"Expected metadatas, got {result.metadatas}"
        )

    def test_query_returns_cosine_ranked_results(self) -> None:
        """
        Given an InMemoryVectorStore with 3 documents at different embedding angles
        When query() is called with an embedding closest to document B
        Then the QueryResult lists document B first with the smallest distance
        """
        # Given: 3 documents with different embeddings
        store = InMemoryVectorStore()
        store.add_documents(
            "test",
            ids=["a", "b", "c"],
            documents=["Doc A", "Doc B", "Doc C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        # When: query with VEC_B (closest to doc B)
        result = store.query("test", query_embedding=VEC_B, n_results=3)

        # Then: doc B is first (smallest distance)
        assert isinstance(result, QueryResult), (
            f"query() should return QueryResult, got {type(result).__name__}"
        )
        assert result.ids[0][0] == "b", (
            f"Expected doc 'b' first (closest to VEC_B), got {result.ids[0][0]}"
        )
        assert result.distances[0][0] <= result.distances[0][1], (
            "First distance should be <= second (sorted by similarity)"
        )

    def test_query_n_results_limits_output(self) -> None:
        """
        Given an InMemoryVectorStore with 5 documents
        When query(n_results=2) is called
        Then the QueryResult contains exactly 2 documents
        """
        # Given: 5 documents
        store = InMemoryVectorStore()
        for i in range(5):
            vec = [0.0] * EMBED_DIM
            vec[i % EMBED_DIM] = 1.0
            store.add_documents(
                "test",
                ids=[f"d{i}"],
                documents=[f"Doc {i}"],
                embeddings=[vec],
            )

        # When: query with n_results=2
        result = store.query("test", query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0], n_results=2)

        # Then: exactly 2 results
        assert len(result.ids[0]) == 2, f"Expected 2 results, got {len(result.ids[0])}"

    def test_get_by_metadata_filters_correctly(self) -> None:
        """
        Given an InMemoryVectorStore with documents having different metadata
        When get_by_metadata(where={"verdict": "yes"}) is called
        Then only documents matching the filter are returned
        """
        # Given: documents with different verdicts
        store = InMemoryVectorStore()
        store.add_documents(
            "test",
            ids=["d1", "d2", "d3"],
            documents=["A", "B", "C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
            metadatas=[{"verdict": "yes"}, {"verdict": "no"}, {"verdict": "yes"}],
        )

        # When: filter by verdict=yes
        result = store.get_by_metadata("test", where={"verdict": "yes"})

        # Then: only matching documents returned
        assert isinstance(result, GetResult), (
            f"get_by_metadata() should return GetResult, got {type(result).__name__}"
        )
        assert len(result.ids) == 2, f"Expected 2 matching docs, got {len(result.ids)}"
        assert set(result.ids) == {"d1", "d3"}, f"Expected ids d1 and d3, got {result.ids}"

    def test_get_all_documents_returns_everything(self) -> None:
        """
        Given an InMemoryVectorStore with 3 documents
        When get_all_documents() is called
        Then the GetResult contains all 3 documents
        """
        # Given: 3 documents
        store = InMemoryVectorStore()
        store.add_documents(
            "test",
            ids=["d1", "d2", "d3"],
            documents=["A", "B", "C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        # When: get all
        result = store.get_all_documents("test")

        # Then: all 3 returned
        assert isinstance(result, GetResult), (
            f"get_all_documents() should return GetResult, got {type(result).__name__}"
        )
        assert len(result.ids) == 3, f"Expected 3 documents, got {len(result.ids)}"

    def test_delete_by_id_removes_documents(self) -> None:
        """
        Given an InMemoryVectorStore with documents
        When delete_by_id() is called with one ID
        Then that document is no longer returned by get_documents()
             and collection_count() decreases by 1
        """
        # Given: 3 documents
        store = InMemoryVectorStore()
        store.add_documents(
            "test",
            ids=["d1", "d2", "d3"],
            documents=["A", "B", "C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        # When: delete one
        store.delete_by_id("test", ids=["d2"])

        # Then: deleted doc is gone, count decreased
        assert store.collection_count("test") == 2, (
            f"Expected count=2 after delete, got {store.collection_count('test')}"
        )
        result = store.get_documents("test", ids=["d1", "d3"])
        assert set(result.ids) == {"d1", "d3"}, (
            f"Expected remaining ids d1 and d3, got {result.ids}"
        )

    def test_collection_count_reflects_stored_documents(self) -> None:
        """
        Given an InMemoryVectorStore with N documents in a collection
        When collection_count() is called
        Then it returns N
        """
        # Given: 3 documents
        store = InMemoryVectorStore()
        store.add_documents(
            "test",
            ids=["d1", "d2", "d3"],
            documents=["A", "B", "C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        # When/Then: count is 3
        assert store.collection_count("test") == 3, (
            f"Expected count=3, got {store.collection_count('test')}"
        )

    def test_reset_collection_drops_all_documents(self) -> None:
        """
        Given an InMemoryVectorStore with documents in a collection
        When reset_collection() is called
        Then collection_count() returns 0
        """
        # Given: 3 documents
        store = InMemoryVectorStore()
        store.add_documents(
            "test",
            ids=["d1", "d2", "d3"],
            documents=["A", "B", "C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        # When: reset
        store.reset_collection("test")

        # Then: count is 0
        assert store.collection_count("test") == 0, (
            f"Expected count=0 after reset, got {store.collection_count('test')}"
        )

    def test_nonexistent_collection_raises_actionable_error(self) -> None:
        """
        Given an InMemoryVectorStore with no collections
        When get_documents() is called with a collection name
        Then it raises ActionableError of type INDEX
        """
        # Given: empty store
        store = InMemoryVectorStore()

        # When/Then: accessing non-existent collection raises
        with pytest.raises(ActionableError) as exc_info:
            store.get_documents("nonexistent", ids=["x"])

        assert exc_info.value.error_type == "index", (
            f"Expected INDEX error type, got {exc_info.value.error_type}"
        )

    def test_add_documents_upserts_existing_ids(self) -> None:
        """
        Given an InMemoryVectorStore with a document at ID "doc-1"
        When add_documents() is called with the same ID but different content
        Then get_documents() returns the updated content
             and collection_count() is unchanged
        """
        # Given: a document
        store = InMemoryVectorStore()
        store.add_documents(
            "test",
            ids=["doc-1"],
            documents=["Original"],
            embeddings=[VEC_A],
            metadatas=[{"v": "1"}],
        )

        # When: upsert with same ID
        store.add_documents(
            "test",
            ids=["doc-1"],
            documents=["Updated"],
            embeddings=[VEC_B],
            metadatas=[{"v": "2"}],
        )

        # Then: content is updated, count unchanged
        assert store.collection_count("test") == 1, (
            f"Expected count=1 after upsert, got {store.collection_count('test')}"
        )
        result = store.get_documents("test", ids=["doc-1"])
        assert result.documents == ["Updated"], f"Expected updated content, got {result.documents}"
        assert result.metadatas == [{"v": "2"}], (
            f"Expected updated metadata, got {result.metadatas}"
        )

    def test_close_is_noop(self) -> None:
        """
        Given an InMemoryVectorStore
        When close() is called
        Then no error is raised and the store remains usable
        """
        # Given: a store with data
        store = InMemoryVectorStore()
        store.add_documents("test", ids=["d1"], documents=["A"], embeddings=[VEC_A])

        # When: close and then use
        store.close()
        count = store.collection_count("test")

        # Then: still usable
        assert count == 1, f"Expected count=1 after close, got {count}"

    def test_writes_immediately_visible(self) -> None:
        """
        Given an InMemoryVectorStore shared by two references
        When one reference adds a document
        Then the other reference sees it immediately via get_documents()
        """
        # Given: two references to the same store
        store = InMemoryVectorStore()
        ref1 = store
        ref2 = store

        # When: one reference adds a document
        ref1.add_documents("test", ids=["d1"], documents=["Visible"], embeddings=[VEC_A])

        # Then: the other reference sees it immediately
        result = ref2.get_documents("test", ids=["d1"])
        assert result.documents == ["Visible"], (
            f"Expected immediate visibility, got {result.documents}"
        )

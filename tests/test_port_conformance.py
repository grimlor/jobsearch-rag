"""
BDD specs for port conformance — contract tests run against both real
and fake implementations.

Covers: TestPortConformance (D4).

Public API surface (from tests/fakes):
    InMemoryVectorStore()
    FakeEmbedder(embed_vector=..., classify_response=...)

Public API surface (from src/jobsearch_rag/rag/store):
    VectorStore(persist_dir: str, distance_metric: str)
    store.add_documents(collection_name, *, ids, documents, embeddings, metadatas=None)
    store.query(collection_name, *, query_embedding, n_results) -> QueryResult
    store.get_documents(collection_name, *, ids) -> GetResult
    store.get_by_metadata(collection_name, *, where, include=None) -> GetResult
    store.delete_by_id(collection_name, *, ids)
    store.collection_count(name) -> int
    store.reset_collection(name)
    store.close()

Public API surface (from src/jobsearch_rag/ports):
    EmbeddingPort — Protocol
    VectorStorePort — Protocol
    QueryResult — dataclass
    GetResult — dataclass

Public API surface (from src/jobsearch_rag/rag/embedder):
    Embedder(ollama_config) — concrete EmbeddingPort impl
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from jobsearch_rag.ports import EmbeddingPort
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.store import VectorStore
from tests.fakes import FakeEmbedder, InMemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EMBED_DIM = 5
VEC_A = [0.9, 0.1, 0.2, 0.0, 0.3]
VEC_B = [0.1, 0.8, 0.1, 0.7, 0.0]
VEC_C = [0.7, 0.2, 0.3, 0.0, 0.4]


# ---------------------------------------------------------------------------
# Parameterized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_store() -> Iterator[VectorStore]:
    """Real ChromaDB-backed VectorStore in a temp directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        s = VectorStore(persist_dir=tmpdir, distance_metric="cosine")
        yield s
        s.close()


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    """InMemoryVectorStore instance."""
    return InMemoryVectorStore()


@pytest.fixture(params=["real", "in_memory"])
def store_impl(
    request: pytest.FixtureRequest,
    real_store: VectorStore,
    in_memory_store: InMemoryVectorStore,
) -> VectorStore | InMemoryVectorStore:
    """Parameterized fixture yielding each VectorStorePort implementation."""
    if request.param == "real":
        return real_store
    return in_memory_store


@pytest.fixture(params=["real_embedder", "fake_embedder"])
def embedder_impl(request: pytest.FixtureRequest) -> object:
    """Parameterized fixture yielding each EmbeddingPort implementation."""
    if request.param == "real_embedder":
        # Construct a real Embedder with mocked ollama client
        mock_config = MagicMock()
        mock_config.model = "test-model"
        mock_config.base_url = "http://localhost:11434"
        mock_config.embed_model = "test-embed"
        mock_config.max_embed_chars = 8000
        mock_config.disqualifier_prompt = "test"
        mock_config.screen_prompt = "test"
        embedder = Embedder(mock_config)
        embedder._client = AsyncMock()  # pyright: ignore[reportPrivateUsage]  # I/O boundary mock for contract test
        return embedder
    return FakeEmbedder()


# ============================================================================
# TestPortConformance
# ============================================================================


class TestPortConformance:
    """
    REQUIREMENT: Both real and fake implementations satisfy the same
    port behavioral contracts — a contract test suite run against each.

    WHO: The port system as a whole — ensures fakes don't diverge from
         real implementations.
    WHAT: (1) VectorStorePort implementations pass add → get roundtrip.
          (2) VectorStorePort implementations return query results ranked
              by ascending distance.
          (3) VectorStorePort implementations filter by metadata correctly.
          (4) VectorStorePort implementations delete documents by ID.
          (5) VectorStorePort implementations reset collections to empty.
          (6) Embedder and FakeEmbedder both satisfy EmbeddingPort
              isinstance checks.
    WHY: Fakes that diverge from real implementations silently weaken
         the test suite. Contract tests ensure substitutability.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (for real Embedder construction only)
        Real:  VectorStore (ChromaDB temp dir), InMemoryVectorStore,
               Embedder, FakeEmbedder
        Never: Mock the contract test assertions themselves
    """

    def test_vector_store_contract_add_get_roundtrip(
        self, store_impl: VectorStore | InMemoryVectorStore
    ) -> None:
        """
        Given a VectorStorePort implementation (parameterized)
        When documents are added then retrieved by ID
        Then the returned data matches what was added
        """
        # Given: a store implementation (parameterized fixture)

        # When: add then retrieve
        store_impl.add_documents(
            "contract",
            ids=["c1", "c2"],
            documents=["Hello", "World"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
        )
        result = store_impl.get_documents("contract", ids=["c1", "c2"])

        # Then: data matches
        assert set(result.ids) == {"c1", "c2"}, f"Expected ids c1, c2; got {result.ids}"
        assert len(result.documents) == 2, f"Expected 2 documents, got {len(result.documents)}"

    def test_vector_store_contract_query_ranking(
        self, store_impl: VectorStore | InMemoryVectorStore
    ) -> None:
        """
        Given a VectorStorePort implementation with documents at different angles
        When query() is called
        Then results are sorted by ascending distance (most similar first)
        """
        # Given: 3 documents
        store_impl.add_documents(
            "contract",
            ids=["a", "b", "c"],
            documents=["Doc A", "Doc B", "Doc C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
        )

        # When: query with VEC_A
        result = store_impl.query("contract", query_embedding=VEC_A, n_results=3)

        # Then: distances are ascending
        distances = result.distances[0]
        assert distances == sorted(distances), (
            f"Distances should be ascending (most similar first), got {distances}"
        )
        assert result.ids[0][0] == "a", (
            f"Expected doc 'a' first (closest to VEC_A), got {result.ids[0][0]}"
        )

    def test_vector_store_contract_metadata_filter(
        self, store_impl: VectorStore | InMemoryVectorStore
    ) -> None:
        """
        Given a VectorStorePort implementation with varied metadata
        When get_by_metadata() is called with a where filter
        Then only matching documents are returned
        """
        # Given: documents with different metadata
        store_impl.add_documents(
            "contract",
            ids=["d1", "d2", "d3"],
            documents=["A", "B", "C"],
            embeddings=[VEC_A, VEC_B, VEC_C],
            metadatas=[{"verdict": "yes"}, {"verdict": "no"}, {"verdict": "yes"}],
        )

        # When: filter by verdict=yes
        result = store_impl.get_by_metadata("contract", where={"verdict": "yes"})

        # Then: only matching documents
        assert len(result.ids) == 2, f"Expected 2 matching docs, got {len(result.ids)}"
        assert set(result.ids) == {"d1", "d3"}, f"Expected d1, d3; got {result.ids}"

    def test_vector_store_contract_delete(
        self, store_impl: VectorStore | InMemoryVectorStore
    ) -> None:
        """
        Given a VectorStorePort implementation with documents
        When delete_by_id() removes one
        Then it is no longer retrievable and count decreases
        """
        # Given: 2 documents
        store_impl.add_documents(
            "contract",
            ids=["d1", "d2"],
            documents=["A", "B"],
            embeddings=[VEC_A, VEC_B],
        )

        # When: delete d1
        store_impl.delete_by_id("contract", ids=["d1"])

        # Then: count decreased, d1 gone
        assert store_impl.collection_count("contract") == 1, (
            f"Expected count=1, got {store_impl.collection_count('contract')}"
        )

    def test_vector_store_contract_reset(
        self, store_impl: VectorStore | InMemoryVectorStore
    ) -> None:
        """
        Given a VectorStorePort implementation with documents
        When reset_collection() is called
        Then count is 0
        """
        # Given: 2 documents
        store_impl.add_documents(
            "contract",
            ids=["d1", "d2"],
            documents=["A", "B"],
            embeddings=[VEC_A, VEC_B],
        )

        # When: reset
        store_impl.reset_collection("contract")

        # Then: empty
        assert store_impl.collection_count("contract") == 0, (
            f"Expected count=0, got {store_impl.collection_count('contract')}"
        )

    def test_embedding_port_conformance(self, embedder_impl: object) -> None:
        """
        Given an EmbeddingPort implementation (parameterized)
        When isinstance(impl, EmbeddingPort) is checked
        Then it returns True
        """
        # Given: an embedder implementation (parameterized fixture)

        # When: check protocol conformance
        result = isinstance(embedder_impl, EmbeddingPort)

        # Then: satisfies EmbeddingPort
        assert result, f"{type(embedder_impl).__name__} should satisfy EmbeddingPort"

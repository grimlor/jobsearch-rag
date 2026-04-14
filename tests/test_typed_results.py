"""
BDD specs for D3 -- typed results + eval abstraction leak fix.

Covers: TestVectorStoreReturnsTypedResults (5 tests),
        TestEvalRunnerAbstractionLeak (3 tests).

Public API surface (from src/jobsearch_rag/rag/store):
    VectorStore(persist_dir: str, distance_metric: str)
    store.query(collection_name, *, query_embedding, n_results) -> QueryResult
    store.get_documents(collection_name, *, ids) -> GetResult
    store.get_by_metadata(collection_name, *, where, include) -> GetResult
    store.get_all_documents(collection_name, *, include) -> GetResult
    store.add_documents(collection_name, *, ids, documents, embeddings, metadatas)
    store.collection_count(name) -> int
    store.reset_collection(name)
    store.close()

Public API surface (from src/jobsearch_rag/ports):
    QueryResult  -- dataclass (ids, documents, metadatas, distances)
    GetResult    -- dataclass (ids, documents, metadatas)

Public API surface (from src/jobsearch_rag/pipeline/eval):
    EvalRunner(scorer, ranker, store)
    eval_runner.evaluate() -> EvalResult
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import jobsearch_rag.pipeline.eval as eval_module
import tests.fakes as fakes_mod
from jobsearch_rag.pipeline.eval import EvalRunner
from jobsearch_rag.ports import GetResult, QueryResult
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EMBED_DIM = 5
EMBED_A = [0.9, 0.1, 0.2, 0.0, 0.3]
EMBED_B = [0.1, 0.8, 0.1, 0.7, 0.0]
EMBED_C = [0.7, 0.2, 0.3, 0.0, 0.4]


@pytest.fixture
def store() -> Iterator[VectorStore]:
    """VectorStore backed by a temp directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        s = VectorStore(persist_dir=tmpdir, distance_metric="cosine")
        yield s
        s.close()


@pytest.fixture
def populated_store(store: VectorStore) -> VectorStore:
    """VectorStore with 3 documents pre-loaded in 'test_collection'."""
    store.add_documents(
        collection_name="test_collection",
        ids=["doc-1", "doc-2", "doc-3"],
        documents=["Alpha document", "Beta document", "Gamma document"],
        embeddings=[EMBED_A, EMBED_B, EMBED_C],
        metadatas=[
            {"verdict": "yes", "job_id": "j1"},
            {"verdict": "no", "job_id": "j2"},
            {"verdict": "yes", "job_id": "j3"},
        ],
    )
    return store


class TestVectorStoreReturnsTypedResults:
    """
    REQUIREMENT: VectorStore methods return QueryResult/GetResult
    dataclasses instead of dict[str, Any].

    WHO: All domain callers of VectorStore (Scorer, DecisionRecorder,
         Indexer, EvalRunner) -- they receive typed results.
    WHAT: (1) VectorStore.query() returns a QueryResult instance.
          (2) VectorStore.get_documents() returns a GetResult instance.
          (3) VectorStore.get_by_metadata() returns a GetResult instance.
          (4) VectorStore.get_all_documents() returns a GetResult instance.
          (5) Querying an empty collection returns an empty QueryResult.
    WHY: Typed returns make the port contract verifiable by pyright.
         Eliminates untyped string-key dict access across the codebase.

    MOCK BOUNDARY:
        Mock:  Nothing -- uses real ChromaDB in temp dir
        Real:  VectorStore (full ChromaDB stack), QueryResult, GetResult
        Never: Mock VectorStore internals; never construct results manually
    """

    def test_query_returns_query_result(self, populated_store: VectorStore) -> None:
        """
        Given a VectorStore with documents in a collection
        When store.query() is called
        Then the return value is a QueryResult instance with populated fields
        """
        # Given: a populated store (fixture)

        # When: query is called
        result = populated_store.query(
            "test_collection",
            query_embedding=EMBED_A,
            n_results=2,
        )

        # Then: return value is a QueryResult with populated fields
        assert isinstance(result, QueryResult), (
            f"query() should return QueryResult, got {type(result).__name__}"
        )
        assert len(result.ids[0]) == 2, f"Expected 2 result IDs, got {len(result.ids[0])}"
        assert len(result.distances[0]) == 2, (
            f"Expected 2 distances, got {len(result.distances[0])}"
        )

    def test_get_documents_returns_get_result(self, populated_store: VectorStore) -> None:
        """
        Given a VectorStore with documents in a collection
        When store.get_documents() is called with known IDs
        Then the return value is a GetResult instance with matching data
        """
        # Given: a populated store (fixture)

        # When: get_documents is called
        result = populated_store.get_documents("test_collection", ids=["doc-1"])

        # Then: return value is a GetResult with matching data
        assert isinstance(result, GetResult), (
            f"get_documents() should return GetResult, got {type(result).__name__}"
        )
        assert result.ids == ["doc-1"], f"Expected ids=['doc-1'], got {result.ids}"

    def test_get_by_metadata_returns_get_result(self, populated_store: VectorStore) -> None:
        """
        Given a VectorStore with documents containing metadata
        When store.get_by_metadata() is called with a where filter
        Then the return value is a GetResult instance with matching data
        """
        # Given: a populated store (fixture)

        # When: get_by_metadata filters by verdict
        result = populated_store.get_by_metadata(
            "test_collection",
            where={"verdict": "yes"},
            include=["metadatas"],
        )

        # Then: return value is a GetResult with matching entries
        assert isinstance(result, GetResult), (
            f"get_by_metadata() should return GetResult, got {type(result).__name__}"
        )
        assert len(result.ids) == 2, (
            f"Expected 2 matching docs (verdict=yes), got {len(result.ids)}"
        )

    def test_get_all_documents_returns_get_result(self, populated_store: VectorStore) -> None:
        """
        Given a VectorStore with documents in a collection
        When store.get_all_documents() is called
        Then the return value is a GetResult instance containing all documents
        """
        # Given: a populated store (fixture)

        # When: get_all_documents is called
        result = populated_store.get_all_documents(
            "test_collection",
            include=["documents", "metadatas"],
        )

        # Then: return value is a GetResult with all 3 documents
        assert isinstance(result, GetResult), (
            f"get_all_documents() should return GetResult, got {type(result).__name__}"
        )
        assert len(result.ids) == 3, f"Expected 3 documents, got {len(result.ids)}"

    def test_query_empty_collection_returns_empty_query_result(self, store: VectorStore) -> None:
        """
        Given a VectorStore with an empty collection
        When store.query() is called
        Then a QueryResult is returned with empty nested lists
        """
        # Given: an empty collection
        store.get_or_create_collection("empty_collection")

        # When: query against empty collection
        result = store.query(
            "empty_collection",
            query_embedding=[0.0] * EMBED_DIM,
            n_results=5,
        )

        # Then: empty QueryResult
        assert isinstance(result, QueryResult), (
            f"Empty query should return QueryResult, got {type(result).__name__}"
        )
        assert result.ids == [[]], f"Empty query ids should be [[]], got {result.ids}"


class TestEvalRunnerAbstractionLeak:
    """
    REQUIREMENT: EvalRunner loads decisions through the VectorStore port
    (get_all_documents) instead of bypassing it via raw chromadb.Collection.

    WHO: EvalRunner -- the only domain class that previously reached through
         VectorStore to the underlying ChromaDB collection.
    WHAT: (1) evaluate() correctly loads and scores decisions stored in the
              VectorStore, proving the internal pipeline reads through the port.
          (2) No chromadb import exists in pipeline/eval.py.
          (3) The eval pipeline works identically with InMemoryVectorStore
              as with real VectorStore (D4).
    WHY: get_or_create_collection() returned a raw chromadb.Collection,
         bypassing the port boundary. This prevents substitution with
         non-ChromaDB backends and makes eval untestable without ChromaDB.

    MOCK BOUNDARY:
        Mock:  Scorer (Ollama I/O boundary), Ranker (deterministic stub)
        Real:  EvalRunner, VectorStore, GetResult
        Never: Mock VectorStore methods; never import chromadb in eval.py
    """

    async def test_evaluate_loads_decisions_through_port(
        self, populated_store: VectorStore
    ) -> None:
        """
        Given an EvalRunner with decisions stored in a VectorStore
        When evaluate() is called
        Then it returns an EvalResult reflecting the stored decisions,
             proving decisions are loaded through the port boundary
        """
        # Given: a store with decisions seeded, plus scorer/ranker stubs
        populated_store.add_documents(
            "decisions",
            ids=["dec-1"],
            documents=["Some JD text"],
            embeddings=[[0.1] * EMBED_DIM],
            metadatas=[{"job_id": "j1", "verdict": "yes"}],
        )

        scorer_stub = AsyncMock()
        scorer_stub.score = AsyncMock(
            return_value=MagicMock(
                fit_score=0.5,
                archetype_score=0.5,
                history_score=0.5,
                comp_score=0.5,
                negative_score=0.0,
                culture_score=0.5,
                disqualified=False,
                disqualify_reason=None,
                best_archetype="Test",
                explanation="test",
            )
        )
        ranker_stub = MagicMock()
        ranker_stub.min_score_threshold = 0.4
        ranker_stub.compute_final_score = MagicMock(return_value=0.6)

        runner = EvalRunner(scorer=scorer_stub, ranker=ranker_stub, store=populated_store)

        # When: evaluate is called (exercises _load_decisions internally)
        result = await runner.evaluate()

        # Then: EvalResult reflects the 1 stored decision loaded through the port
        assert result.decisions_evaluated == 1, (
            f"Expected 1 decision evaluated, got {result.decisions_evaluated}"
        )
        assert len(result.per_decision) == 1, (
            f"Expected 1 per-decision entry, got {len(result.per_decision)}"
        )
        assert result.per_decision[0].job_id == "j1", (
            f"Expected job_id='j1', got {result.per_decision[0].job_id!r}"
        )
        assert result.per_decision[0].verdict == "yes", (
            f"Expected verdict='yes', got {result.per_decision[0].verdict!r}"
        )

    async def test_eval_pipeline_works_with_in_memory_store(self) -> None:
        """
        Given an EvalRunner constructed with InMemoryVectorStore
        When evaluate() is called with decisions in the store
        Then it returns valid EvalResult without ChromaDB dependency
        """
        # Given: an InMemoryVectorStore with decisions
        store = fakes_mod.InMemoryVectorStore()
        store.add_documents(
            "decisions",
            ids=["dec-1"],
            documents=["JD text for in-memory test"],
            embeddings=[[0.1] * EMBED_DIM],
            metadatas=[{"job_id": "mem-j1", "verdict": "yes"}],
        )

        scorer_stub = AsyncMock()
        scorer_stub.score = AsyncMock(
            return_value=MagicMock(
                fit_score=0.5,
                archetype_score=0.5,
                history_score=0.5,
                comp_score=0.5,
                negative_score=0.0,
                culture_score=0.5,
                disqualified=False,
                disqualify_reason=None,
                best_archetype="Test",
                explanation="test",
            )
        )
        ranker_stub = MagicMock()
        ranker_stub.min_score_threshold = 0.4
        ranker_stub.compute_final_score = MagicMock(return_value=0.6)

        # cast until D5 changes EvalRunner.__init__ to accept VectorStorePort
        runner = EvalRunner(
            scorer=scorer_stub,
            ranker=ranker_stub,
            store=cast("VectorStore", store),
        )

        # When: evaluate
        result = await runner.evaluate()

        # Then: EvalResult reflects the decision -- no ChromaDB needed
        assert result.decisions_evaluated == 1, (
            f"Expected 1 decision, got {result.decisions_evaluated}"
        )
        assert result.per_decision[0].job_id == "mem-j1", (
            f"Expected job_id='mem-j1', got {result.per_decision[0].job_id!r}"
        )

    def test_eval_module_has_no_chromadb_import(self) -> None:
        """
        Given the source file src/jobsearch_rag/pipeline/eval.py
        When its imports are inspected
        Then 'chromadb' does not appear in any import statement
        """
        # Given: the eval module
        # When: locate the source file
        source_file = importlib.util.find_spec(eval_module.__name__)
        assert source_file is not None and source_file.origin is not None, (
            "Could not locate eval.py source file"
        )
        source_text = Path(source_file.origin).read_text(encoding="utf-8")

        # Then: no chromadb import
        import_lines = [
            line.strip()
            for line in source_text.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        chromadb_imports = [line for line in import_lines if "chromadb" in line]
        assert chromadb_imports == [], (
            f"eval.py should not import chromadb. Found: {chromadb_imports}"
        )

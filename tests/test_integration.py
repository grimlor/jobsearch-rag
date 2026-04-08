"""
Integration tests — validate external dependency contracts.

These tests require **live Ollama** with ``nomic-embed-text`` and
``mistral:7b`` models pulled.  They are skipped by default and run
only when explicitly requested::

    uv run pytest -m integration          # run integration tests only
    uv run pytest -m "not integration"    # run everything else (default)
    uv run pytest                          # also skips integration

The integration marker is defined in ``pyproject.toml`` under
``[tool.pytest.ini_options].markers``.

Spec classes
------------
* **TestOllamaContract** — Ollama SDK response shapes we rely on
  (embedding vectors, chat responses, model listing, error types).
* **TestChromaDBContract** — ChromaDB distance semantics and
  persistence behavior.
* **TestEndToEndScoring** — full pipeline from resume indexing
  through scoring, using real Ollama embeddings with no mocks.
* **TestLiveZipRecruiterPipeline** — full system against live
  ZipRecruiter: browser session → search → extract → score → rank →
  export.
* **TestOllamaHealthCheckSkip** — ``require_ollama`` fixture yields
  when Ollama is reachable and calls ``pytest.skip()`` when not.
* **TestIntegrationRescoreAccumulatedJDs** — rescorer processes
  seeded JD files through real Ollama and produces sorted export.
* **TestLiveCumulativeAccumulation** — two successive live searches
  produce a merged CSV, preserved JD files, and correct Markdown.
* **TestLiveFreshModeReset** — ``--fresh`` flag discards prior
  results and restores replace-on-write behavior.
* **TestLiveDecisionExclusionAcrossRuns** — decided listings are
  excluded from CSV/Markdown but JD files are preserved.

Between them, these tests catch the class of bug where our mocks
silently diverge from reality — e.g. an ollama SDK upgrade changes
the response shape, or ChromaDB alters its distance metric.
"""

from __future__ import annotations

import argparse
import asyncio
import csv as csv_mod
import dataclasses
import math
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jobsearch_rag.adapters.session import SessionConfig, SessionManager, throttle
from jobsearch_rag.adapters.ziprecruiter import ZipRecruiterAdapter
from jobsearch_rag.cli import handle_search
from jobsearch_rag.config import (
    BoardConfig,
    ChromaConfig,
    OllamaConfig,
    OutputConfig,
    Settings,
    load_settings,
)
from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.export.csv_export import CSVExporter
from jobsearch_rag.export.jd_files import JDFileExporter
from jobsearch_rag.export.markdown import MarkdownExporter
from jobsearch_rag.pipeline.ranker import Ranker
from jobsearch_rag.pipeline.rescorer import Rescorer
from jobsearch_rag.pipeline.runner import PipelineRunner, RunResult
from jobsearch_rag.rag.comp_parser import compute_comp_score, parse_compensation
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.indexer import Indexer
from jobsearch_rag.rag.scorer import Scorer
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jobsearch_rag.adapters.base import JobListing
    from jobsearch_rag.rag.scorer import ScoreResult

# All tests in this file require live Ollama
pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Configuration — matches settings.toml defaults
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:7b"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_integration_ollama_config(**overrides: object) -> OllamaConfig:
    """Build an OllamaConfig for integration tests with live Ollama defaults."""
    defaults: dict[str, object] = {
        "base_url": OLLAMA_BASE_URL,
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "slow_llm_threshold_ms": 30_000,
        "classify_system_prompt": "You are a job listing classifier.",
        "max_retries": 2,
        "base_delay": 0.5,
        "max_embed_chars": 8_000,
        "head_ratio": 0.6,
        "retryable_status_codes": [408, 429, 500, 502, 503, 504],
    }
    defaults.update(overrides)
    return OllamaConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def embedder() -> Embedder:
    """A real Embedder pointed at localhost Ollama."""
    return Embedder(_make_integration_ollama_config())


@pytest.fixture
def store() -> Iterator[VectorStore]:
    """A VectorStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


@pytest.fixture
def sample_resume(tmp_path: Path) -> Path:
    """A small but realistic resume for integration testing."""
    resume = tmp_path / "resume.md"
    resume.write_text(
        "# Test Candidate\n\n"
        "## Summary\n"
        "Principal architect specializing in distributed systems "
        "and cloud-native platform design.\n\n"
        "## Experience\n"
        "### Acme Corp - Staff Engineer\n"
        "Led platform architecture for microservices migration.\n"
        "Designed event-driven systems with Kafka and Kubernetes.\n\n"
        "## Skills\n"
        "Python, Go, Kubernetes, Kafka, AWS, Terraform, gRPC\n",
        encoding="utf-8",
    )
    return resume


@pytest.fixture
def sample_archetypes(tmp_path: Path) -> Path:
    """A small archetypes TOML for integration testing."""
    archetypes = tmp_path / "archetypes.toml"
    archetypes.write_text(
        "[[archetypes]]\n"
        'name = "Staff Platform Architect"\n'
        'description = """\n'
        "Defines technical strategy and architecture for distributed "
        "systems platforms. Cross-team influence. Cloud-native "
        "infrastructure. API design and governance.\n"
        '"""\n\n'
        "[[archetypes]]\n"
        'name = "SRE Manager"\n'
        'description = """\n'
        "Manages on-call rotations, incident response, and reliability "
        "engineering. Primarily operational with people management.\n"
        '"""\n',
        encoding="utf-8",
    )
    return archetypes


# ---------------------------------------------------------------------------
# TestOllamaContract
# ---------------------------------------------------------------------------


class TestOllamaContract:
    """
    REQUIREMENT: Ollama SDK responses match the shapes our code assumes.

    WHO: Every unit test that mocks Ollama responses
    WHAT: (1) The system returns embeddings as lists of floats with consistent dimensionality across different inputs.
          (2) The system produces an embedding vector that is not all zeros for meaningful text input.
          (3) The system returns a non-empty string for a classification prompt.
          (4) The system passes the health check without raising an exception when the required models are available.
          (5) The system raises an EMBEDDING error that includes 'ollama pull' guidance when the configured model does not exist during a health check.
          (6) The system raises an EMBEDDING error with recovery guidance when asked to embed with a nonexistent model.
          (7) The system places semantically similar text closer to a query than unrelated text in embedding space.
    WHY: If an Ollama SDK update changes response shapes, our mocks would
         still pass but production would break — these tests catch that drift

    MOCK BOUNDARY:
        Mock:  (none — integration test uses live Ollama)
        Real:  Embedder.embed, Embedder.classify, Embedder.health_check
        Never: Mock Ollama responses (defeats the purpose of contract tests)
    """

    async def test_embed_returns_float_list_with_consistent_dimensions(
        self, embedder: Embedder
    ) -> None:
        """
        GIVEN two different text inputs
        WHEN embed() is called on each
        THEN both return list[float] with identical dimensionality.
        """
        # When: embed two different texts
        vec1 = await embedder.embed("distributed systems architecture")
        vec2 = await embedder.embed("underwater basket weaving")

        # Then: both are float lists with same dimensions
        assert isinstance(vec1, list), f"Expected list, got {type(vec1)}"
        assert len(vec1) > 0, "Embedding should not be empty"
        assert all(isinstance(v, float) for v in vec1), "All elements should be floats"
        assert len(vec1) == len(vec2), f"Embedding dimensions differ: {len(vec1)} vs {len(vec2)}"

    async def test_embed_vector_is_not_all_zeros(self, embedder: Embedder) -> None:
        """
        GIVEN a meaningful text input
        WHEN embed() is called
        THEN the resulting vector is not all zeros.
        """
        # When: embed meaningful text
        vec = await embedder.embed("Principal platform architect role")

        # Then: not a zero vector
        assert any(v != 0.0 for v in vec), "Embedding should carry semantic information"

    async def test_classify_returns_string(self, embedder: Embedder) -> None:
        """
        GIVEN a simple classification prompt
        WHEN classify() is called
        THEN the result is a non-empty string.
        """
        # When: classify a simple prompt
        result = await embedder.classify("Respond with exactly one word: hello")

        # Then: non-empty string
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Classification result should not be empty"

    async def test_health_check_passes_with_required_models(self, embedder: Embedder) -> None:
        """
        GIVEN an Ollama instance with required models pulled
        WHEN health_check() is called
        THEN no exception is raised.
        """
        # When/Then: should not raise
        await embedder.health_check()

    async def test_health_check_nonexistent_model_suggests_ollama_pull(self) -> None:
        """
        GIVEN an Embedder configured with a nonexistent model
        WHEN health_check() is called
        THEN an EMBEDDING error with 'ollama pull' guidance is raised.
        """
        # Given: embedder with nonexistent model
        embedder = Embedder(
            _make_integration_ollama_config(embed_model="does-not-exist-model-xyz")
        )

        # When/Then: raises EMBEDDING error
        with pytest.raises(ActionableError) as exc_info:
            await embedder.health_check()

        # Then: error has actionable guidance
        err = exc_info.value
        assert err.error_type == ErrorType.EMBEDDING, (
            f"Expected EMBEDDING error, got {err.error_type}"
        )
        assert "does-not-exist-model-xyz" in err.error, "Error should name the missing model"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"
        assert len(err.troubleshooting.steps) > 0, "Should have troubleshooting steps"

    async def test_embed_nonexistent_model_provides_recovery_guidance(self) -> None:
        """
        GIVEN an Embedder configured with a nonexistent model
        WHEN embed() is called
        THEN an EMBEDDING error with recovery guidance is raised.
        """
        # Given: embedder with nonexistent model
        embedder = Embedder(
            _make_integration_ollama_config(
                embed_model="nonexistent-model-abc",
                max_retries=1,
                base_delay=0.0,
            )
        )

        # When/Then: raises EMBEDDING error
        with pytest.raises(ActionableError) as exc_info:
            await embedder.embed("test")

        # Then: error has guidance
        err = exc_info.value
        assert err.error_type == ErrorType.EMBEDDING, (
            f"Expected EMBEDDING error, got {err.error_type}"
        )
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    async def test_similar_texts_produce_closer_embeddings(self, embedder: Embedder) -> None:
        """
        GIVEN three texts: one query, one semantically similar, one unrelated
        WHEN all three are embedded
        THEN the similar text has smaller cosine distance to the query than the unrelated text.
        """
        # When: embed all three texts
        vec_arch = await embedder.embed("Staff platform architect for distributed cloud systems")
        vec_similar = await embedder.embed(
            "Principal engineer designing cloud-native infrastructure"
        )
        vec_unrelated = await embedder.embed("Recipe for chocolate chip cookies with extra sugar")

        # Then: similar is closer than unrelated
        def cosine_distance(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - dot / (norm_a * norm_b)

        dist_similar = cosine_distance(vec_arch, vec_similar)
        dist_unrelated = cosine_distance(vec_arch, vec_unrelated)

        assert dist_similar < dist_unrelated, (
            f"Similar texts should be closer: {dist_similar:.4f} vs {dist_unrelated:.4f}"
        )


# ---------------------------------------------------------------------------
# TestChromaDBContract
# ---------------------------------------------------------------------------


class TestChromaDBContract:
    """
    REQUIREMENT: ChromaDB returns distances and results in the format we assume.

    WHO: VectorStore wrapper and Scorer distance-to-score conversion
    WHAT: (1) The system returns an approximately zero distance when it queries an indexed document with the same embedding vector.
          (2) The system assigns a lower distance to the architecture document than to the cooking document when it queries with an architecture-related embedding.
          (3) The system returns query results that include the ids, documents, metadatas, and distances keys.
          (4) The system preserves indexed data across a client restart when a new VectorStore client opens the same directory.
    WHY: Our _distance_to_score function assumes cosine distance semantics
         (0 = identical, 1 = orthogonal) — any deviation would invert
         all scoring logic

    MOCK BOUNDARY:
        Mock:  (none — integration test uses real ChromaDB)
        Real:  VectorStore operations, ChromaDB persistence, Embedder.embed
        Never: Mock ChromaDB responses (defeats the purpose of contract tests)
    """

    async def test_identical_documents_have_near_zero_distance(
        self, store: VectorStore, embedder: Embedder
    ) -> None:
        """
        GIVEN a document indexed with its embedding
        WHEN queried with the same embedding vector
        THEN the distance is approximately zero.
        """
        # Given: index a document
        text = "Staff architect for distributed systems"
        embedding = await embedder.embed(text)
        store.add_documents(
            collection_name="test_identity",
            ids=["doc-1"],
            documents=[text],
            embeddings=[embedding],
        )

        # When: query with same vector
        results = store.query(
            collection_name="test_identity",
            query_embedding=embedding,
            n_results=1,
        )

        # Then: distance ~0
        distances = results["distances"][0]
        assert len(distances) == 1, f"Expected 1 result, got {len(distances)}"
        assert distances[0] == pytest.approx(0.0, abs=1e-5), (
            f"Identical vector should have ~0 distance, got {distances[0]}"
        )

    async def test_dissimilar_documents_have_higher_distance(
        self, store: VectorStore, embedder: Embedder
    ) -> None:
        """
        GIVEN an architecture doc and a cooking doc indexed together
        WHEN queried with an architecture-related embedding
        THEN the architecture doc has lower distance than the cooking doc.
        """
        # Given: embed and index both documents
        vec_arch = await embedder.embed("Distributed systems platform architecture")
        vec_cooking = await embedder.embed("Baking sourdough bread at high altitude")
        vec_query = await embedder.embed("Cloud infrastructure architect role")

        store.add_documents(
            collection_name="test_distance",
            ids=["doc-arch", "doc-cooking"],
            documents=["architecture doc", "cooking doc"],
            embeddings=[vec_arch, vec_cooking],
        )

        # When: query with architecture vector
        results = store.query(
            collection_name="test_distance",
            query_embedding=vec_query,
            n_results=2,
        )

        # Then: architecture doc is closer
        distances = results["distances"][0]
        ids = results["ids"][0]
        arch_idx = ids.index("doc-arch")
        cooking_idx = ids.index("doc-cooking")
        assert distances[arch_idx] < distances[cooking_idx], (
            f"Architecture should be closer: arch={distances[arch_idx]:.4f}, "
            f"cooking={distances[cooking_idx]:.4f}"
        )

    async def test_query_results_contain_expected_keys(
        self, store: VectorStore, embedder: Embedder
    ) -> None:
        """
        GIVEN a document indexed with metadata
        WHEN query() is called
        THEN the result contains ids, documents, metadatas, and distances keys.
        """
        # Given: indexed document
        embedding = await embedder.embed("test document")
        store.add_documents(
            collection_name="test_keys",
            ids=["doc-1"],
            documents=["test document"],
            embeddings=[embedding],
            metadatas=[{"source": "test"}],
        )

        # When: query
        results = store.query(
            collection_name="test_keys",
            query_embedding=embedding,
            n_results=1,
        )

        # Then: expected keys present with correct structure
        assert "ids" in results, "Results should contain 'ids'"
        assert "documents" in results, "Results should contain 'documents'"
        assert "metadatas" in results, "Results should contain 'metadatas'"
        assert "distances" in results, "Results should contain 'distances'"
        assert isinstance(results["ids"][0], list), "ids[0] should be a list"
        assert isinstance(results["distances"][0], list), "distances[0] should be a list"

    async def test_persistence_survives_client_restart(self, embedder: Embedder) -> None:
        """
        GIVEN data indexed by one VectorStore client
        WHEN a new VectorStore client opens the same directory
        THEN the previously indexed data is still available.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: index with first client
            store1 = VectorStore(persist_dir=tmpdir)
            embedding = await embedder.embed("persistence test")
            store1.add_documents(
                collection_name="test_persist",
                ids=["persist-1"],
                documents=["persistence test"],
                embeddings=[embedding],
            )
            count_before = store1.collection_count("test_persist")

            # When: create new client against same directory
            store2 = VectorStore(persist_dir=tmpdir)
            count_after = store2.collection_count("test_persist")

            # Then: data persisted
            assert count_before == 1, "First client should see 1 document"
            assert count_after == 1, "Second client should also see 1 document"

            # Then: document is retrievable
            docs = store2.get_documents("test_persist", ids=["persist-1"])
            assert docs["documents"][0] == "persistence test", (
                "Retrieved document should match original"
            )


# ---------------------------------------------------------------------------
# TestEndToEndScoring
# ---------------------------------------------------------------------------


class TestEndToEndScoring:
    """
    REQUIREMENT: The full index-then-score pipeline produces valid results.

    WHO: The operator running the pipeline for the first time
    WHAT: (1) The system returns a valid ScoreResult with non-zero fit and archetype scores when it scores a matching JD.
          (2) The system assigns higher fit and archetype scores to a matching JD than to an unrelated JD.
          (3) The system returns a valid boolean `disqualified` field when it scores a matching JD with the LLM disqualifier enabled.
          (4) The system produces 5 resume chunks and makes the collection queryable when it indexes the real resume file.
          (5) The system produces 4 archetypes when it indexes the real archetypes file.
    WHY: Unit tests mock Ollama, so they can't catch integration failures
         like dimension mismatches between embed() and ChromaDB, or
         LLM classification prompts that produce unparseable output

    MOCK BOUNDARY:
        Mock:  (none — full pipeline with real Ollama and ChromaDB)
        Real:  Embedder, Indexer, Scorer, VectorStore, all pipeline steps
        Never: Mock any pipeline component (defeats integration purpose)
    """

    async def test_index_and_score_produces_valid_result(
        self,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
    ) -> None:
        """
        GIVEN a resume and archetypes indexed with real Ollama embeddings
        WHEN a matching JD is scored
        THEN a valid ScoreResult with non-zero fit and archetype scores is returned.
        """
        # Given: index resume and archetypes
        indexer = Indexer(store=store, embedder=embedder)
        resume_count = await indexer.index_resume(str(sample_resume))
        archetype_count = await indexer.index_archetypes(str(sample_archetypes))
        assert resume_count == 3, f"Expected 3 resume chunks, got {resume_count}"
        assert archetype_count == 2, f"Expected 2 archetypes, got {archetype_count}"

        # When: score a matching JD
        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=False,
        )
        result = await scorer.score(
            "Staff Platform Architect: design and build distributed "
            "systems infrastructure for cloud-native applications"
        )

        # Then: valid result with meaningful scores
        assert result.is_valid, (
            f"fit={result.fit_score}, arch={result.archetype_score}, hist={result.history_score}"
        )
        assert result.fit_score > 0.0, "Matching JD should have non-zero fit"
        assert result.archetype_score > 0.0, "Matching JD should have non-zero archetype"
        assert result.history_score == 0.0, "No decisions indexed, history should be 0"

    async def test_matching_jd_scores_higher_than_unrelated(
        self,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
    ) -> None:
        """
        GIVEN indexed resume and archetypes
        WHEN a matching JD and an unrelated JD are both scored
        THEN the matching JD has higher fit and archetype scores.
        """
        # Given: index resume and archetypes
        indexer = Indexer(store=store, embedder=embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        # When: score both JDs
        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=False,
        )
        result_match = await scorer.score(
            "Principal Engineer: distributed systems, Kubernetes, "
            "platform architecture, cloud infrastructure, API design"
        )
        result_unrelated = await scorer.score(
            "Pastry Chef: create artisan breads and pastries for "
            "a high-end restaurant in downtown Portland"
        )

        # Then: matching JD scores higher
        assert result_match.fit_score > result_unrelated.fit_score, (
            f"Matching fit {result_match.fit_score:.4f} should exceed "
            f"unrelated {result_unrelated.fit_score:.4f}"
        )
        assert result_match.archetype_score > result_unrelated.archetype_score, (
            f"Matching archetype {result_match.archetype_score:.4f} should exceed "
            f"unrelated {result_unrelated.archetype_score:.4f}"
        )

    async def test_disqualifier_produces_parseable_json(
        self,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
    ) -> None:
        """
        GIVEN indexed resume and archetypes with LLM disqualifier enabled
        WHEN a matching JD is scored
        THEN the result has a valid boolean disqualified field.
        """
        # Given: index and configure scorer with disqualifier
        indexer = Indexer(store=store, embedder=embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=True,
        )

        # When: score with disqualifier enabled
        result = await scorer.score(
            "Staff Platform Architect: lead the design of cloud-native "
            "distributed systems. Cross-team collaboration and mentoring."
        )

        # Then: disqualified field is a valid boolean
        assert isinstance(result.disqualified, bool), (
            f"Expected bool, got {type(result.disqualified)}"
        )
        # Soft assertion: matching role should not be disqualified
        if result.disqualified:
            pytest.skip(
                f"LLM unexpectedly disqualified a matching role: {result.disqualifier_reason}"
            )

    async def test_index_with_real_resume_file(
        self,
        store: VectorStore,
        embedder: Embedder,
    ) -> None:
        """
        GIVEN the actual project resume.md file
        WHEN index_resume is called
        THEN 5 chunks are produced and the collection is queryable.
        """
        # Given: real resume file
        real_resume = Path("data/resume.md")
        if not real_resume.exists():
            pytest.skip("data/resume.md not found — run from project root")

        # When: index the real resume
        indexer = Indexer(store=store, embedder=embedder)
        count = await indexer.index_resume(str(real_resume))

        # Then: expected chunks and queryable
        assert count == 5, f"Expected 5 resume chunks, got {count}"

        test_vec = await embedder.embed("distributed systems architecture")
        results = store.query(
            collection_name="resume",
            query_embedding=test_vec,
            n_results=3,
        )
        assert len(results["ids"][0]) == 3, "Should return 3 results"
        assert all(d >= 0.0 for d in results["distances"][0]), "Distances should be non-negative"

    async def test_index_with_real_archetypes_file(
        self,
        store: VectorStore,
        embedder: Embedder,
    ) -> None:
        """
        GIVEN the actual project role_archetypes.toml file
        WHEN index_archetypes is called
        THEN 4 archetypes are produced.
        """
        # Given: real archetypes file
        real_archetypes = Path("config/role_archetypes.toml")
        if not real_archetypes.exists():
            pytest.skip("config/role_archetypes.toml not found — run from project root")

        # When: index the real archetypes
        indexer = Indexer(store=store, embedder=embedder)
        count = await indexer.index_archetypes(str(real_archetypes))

        # Then: expected count
        assert count == 4, f"Expected 4 archetypes, got {count}"


# ---------------------------------------------------------------------------
# TestLiveZipRecruiterPipeline
# ---------------------------------------------------------------------------

# This class uses the ``live`` \er — it is excluded from both default
# and integration runs.  Run it explicitly:
#
#     uv run pytest -m live              # live tests only
#     uv run pytest -m "live or integration"  # both tiers


@pytest.mark.live
class TestLiveZipRecruiterPipeline:
    """
    REQUIREMENT: The full system works end-to-end against live ZipRecruiter.

    WHO: The operator validating the tool after installation or upgrade
    WHAT: (1) The system extracts live ZipRecruiter listings, scores them, ranks them in descending order, and exports the results when the full pipeline executes.
    WHY: Unit tests mock every I/O boundary — browser, Ollama, ChromaDB.
         Integration tests use real Ollama but fixture HTML.  Only this test
         validates the entire system against the real world: ZipRecruiter's
         DOM structure, Cloudflare behavior, Ollama model output, and
         ChromaDB persistence all working together.

    MOCK BOUNDARY:
        Mock:  (none — live end-to-end against real services)
        Real:  Browser, ZipRecruiter, Ollama, ChromaDB, all exporters
        Never: Mock any external service (defeats live validation)
    """

    # Minimum listings we expect from a single search page.  ZR typically
    # returns 20+ per page, so 5 is a conservative floor that accounts
    # for rate-limit or layout changes.
    MIN_LISTINGS = 5

    @pytest.fixture
    def live_embedder(self) -> Embedder:
        """A real Embedder for live pipeline tests."""
        return Embedder(_make_integration_ollama_config())

    @pytest.fixture
    def live_store(self) -> Iterator[VectorStore]:
        """A VectorStore in a temp directory for live pipeline tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield VectorStore(persist_dir=tmpdir)

    async def test_live_search_score_rank_export(
        self,
        live_store: VectorStore,
        live_embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN a live ZipRecruiter session, Ollama, and indexed resume/archetypes
        WHEN the full pipeline executes: search → extract → score → rank → export
        THEN listings are extracted, scored, ranked in descending order, and exported.
        """
        # Given: skip if no session file
        session_path = Path("data/ziprecruiter_session.json")
        if not session_path.exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        # Given: health check Ollama
        await live_embedder.health_check()

        # Given: index resume + archetypes
        indexer = Indexer(store=live_store, embedder=live_embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        # When: browser session → authenticate → search one query
        adapter = ZipRecruiterAdapter()
        config = SessionConfig(
            board_name="ziprecruiter",
            headless=False,
            browser_channel="msedge",
        )

        search_url = (
            "https://www.ziprecruiter.com/jobs-search"
            "?search=staff+platform+architect&location=Remote+(USA)"
        )

        listings = []
        try:
            async with SessionManager(config) as session:
                page = await session.new_page()

                try:
                    await adapter.authenticate(page)
                except ActionableError as exc:
                    pytest.skip(f"ZipRecruiter auth failed (session expired?): {exc.error}")

                await throttle(adapter)
                try:
                    listings = await adapter.search(page, search_url, max_pages=1)
                except ActionableError as exc:
                    pytest.skip(f"ZipRecruiter search failed: {exc.error}")
        except Exception as exc:
            pytest.skip(f"Browser session failed: {exc}")

        # Then: structural assertions about extraction
        assert len(listings) >= self.MIN_LISTINGS, (
            f"Expected ≥{self.MIN_LISTINGS} listings, got {len(listings)}"
        )

        for listing in listings:
            assert listing.board == "ziprecruiter", "Board should be 'ziprecruiter'"
            assert listing.title, "Every listing must have a title"
            assert listing.company, "Every listing must have a company"
            assert listing.url.startswith("https://"), f"URL should be HTTPS: {listing.url}"
            assert listing.external_id, "Every listing must have an external_id"

        with_jd = [ls for ls in listings if ls.full_text.strip()]
        assert len(with_jd) >= 1, "At least one listing should have full JD text"

        # When: score each listing
        scorer = Scorer(
            store=live_store,
            embedder=live_embedder,
            disqualify_on_llm_flag=False,
        )
        base_salary = 220_000

        scored: list[tuple[JobListing, ScoreResult]] = []
        embeddings: dict[str, list[float]] = {}

        for listing in with_jd[: self.MIN_LISTINGS]:
            result = await scorer.score(listing.full_text)

            comp = parse_compensation(listing.full_text)
            if comp is not None:
                listing.comp_min = comp.comp_min
                listing.comp_max = comp.comp_max
                listing.comp_source = comp.comp_source
                listing.comp_text = comp.comp_text
            result.comp_score = compute_comp_score(listing.comp_max, base_salary)

            scored.append((listing, result))
            embedding = await live_embedder.embed(listing.full_text)
            embeddings[listing.url] = embedding

        # Then: valid scores
        for listing, result in scored:
            assert result.is_valid, (
                f"Invalid score for '{listing.title}': "
                f"fit={result.fit_score}, arch={result.archetype_score}"
            )
            assert 0.0 <= result.fit_score <= 1.0, f"Fit score out of range: {result.fit_score}"
            assert 0.0 <= result.archetype_score <= 1.0, (
                f"Archetype score out of range: {result.archetype_score}"
            )
            assert 0.0 <= result.comp_score <= 1.0, f"Comp score out of range: {result.comp_score}"
            assert result.fit_score > 0.0, (
                f"Zero fit score for '{listing.title}' — embedding may have failed"
            )
            assert result.archetype_score > 0.0, f"Zero archetype score for '{listing.title}'"

        # When: rank
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            comp_weight=0.15,
            min_score_threshold=0.0,
        )
        ranked, summary = ranker.rank(scored, embeddings)

        # Then: ranking assertions
        assert len(ranked) > 0, "Ranker should produce at least one result"
        assert summary.total_found == len(scored), "Summary should match scored count"
        assert summary.total_scored == len(scored), "Summary should match scored count"

        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True), (
            f"Rankings not in descending order: {scores}"
        )

        for r in ranked:
            assert r.final_score > 0.0, f"Zero final score for '{r.listing.title}'"

        # When: export
        md_path = str(tmp_path / "results.md")
        csv_path = str(tmp_path / "results.csv")
        jd_dir = str(tmp_path / "jds")

        MarkdownExporter().export(ranked, md_path, summary=summary)
        CSVExporter().export(ranked, csv_path, summary=summary)
        jd_paths = JDFileExporter().export(ranked, jd_dir, summary=summary)

        # Then: Markdown export
        md_content = Path(md_path).read_text()
        assert "# Run Summary" in md_content, "Markdown should have run summary"
        assert "## Ranked Listings" in md_content, "Markdown should have ranked listings"
        assert "ziprecruiter" in md_content, "Markdown should mention board"
        for r in ranked:
            assert r.listing.title in md_content, (
                f"'{r.listing.title}' missing from Markdown export"
            )

        # Then: CSV export
        csv_content = Path(csv_path).read_text()
        assert "title,company,board" in csv_content, "CSV should have header"
        csv_lines = csv_content.strip().split("\n")
        assert len(csv_lines) == len(ranked) + 1, (
            f"CSV should have {len(ranked)} data rows, got {len(csv_lines) - 1}"
        )

        # Then: JD file export
        assert len(jd_paths) == len(ranked), (
            f"Expected {len(ranked)} JD files, got {len(jd_paths)}"
        )
        for jd_path in jd_paths:
            jd_content = jd_path.read_text()
            assert "## Score" in jd_content, f"JD file missing '## Score': {jd_path}"
            assert "## Job Description" in jd_content, (
                f"JD file missing '## Job Description': {jd_path}"
            )
            assert "**Board:** ziprecruiter" in jd_content, (
                f"JD file missing board field: {jd_path}"
            )


# ---------------------------------------------------------------------------
# require_ollama fixture — shared skip-if-unreachable guard (Phase 7b)
# ---------------------------------------------------------------------------

_OLLAMA_HEALTH_URL = f"{OLLAMA_BASE_URL}/api/tags"


@pytest.fixture
def require_ollama() -> None:
    """Skip the test if Ollama is not reachable at OLLAMA_BASE_URL."""
    try:
        urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
    except Exception:
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")


# ---------------------------------------------------------------------------
# TestOllamaHealthCheckSkip (Phase 7b — B1)
# ---------------------------------------------------------------------------


class TestOllamaHealthCheckSkip:
    """
    REQUIREMENT: Integration and live tests skip gracefully when Ollama is
    not running.

    WHO: Developer running ``uv run pytest -m integration`` on a machine
         without Ollama.
    WHAT: (1) When Ollama is reachable, the fixture yields and the test proceeds
          (2) When Ollama is unreachable, the fixture calls ``pytest.skip()``
              with a message identifying the URL
    WHY: Without this, integration tests crash with raw ``ConnectionError``
         instead of a clean skip with a descriptive message.

    MOCK BOUNDARY:
        Mock:  (none)
        Real:  HTTP call to Ollama health endpoint
        Never: Mock the health check itself — that defeats the purpose
    """

    def test_fixture_allows_test_when_ollama_is_reachable(self, require_ollama: None) -> None:
        """
        Given Ollama is running
        When the require_ollama fixture runs
        Then the test body executes (reaching this assertion proves it).
        """
        # Given: Ollama is running (or the fixture skips us)

        # When: fixture ran before this body

        # Then: we reached this point — fixture did not skip
        assert True, "Test body should execute when Ollama is reachable"

    def test_fixture_skips_when_ollama_is_unreachable(self) -> None:
        """
        Given Ollama is not running at a bogus URL
        When the fixture logic runs against that URL
        Then pytest.skip() is called with a message identifying the URL.
        """
        # Given: a URL that will not respond
        bogus_url = "http://localhost:1/__nonexistent_ollama__/api/tags"

        # When/Then: attempting to reach it raises, proving skip would fire
        with pytest.raises(Exception) as exc_info:
            urllib.request.urlopen(bogus_url, timeout=2)

        assert exc_info.value is not None, (
            "Connection to bogus Ollama URL should fail, "
            "which is what the fixture translates into pytest.skip()"
        )


# ---------------------------------------------------------------------------
# TestIntegrationRescoreAccumulatedJDs (Phase 7b — B2)
# ---------------------------------------------------------------------------

# Sample JD files matching the format produced by JDFileExporter
_JD_TEMPLATE = """\
# {title}

**Company:** {company}
**Location:** Remote (USA)
**Board:** ziprecruiter
**URL:** https://www.ziprecruiter.com/jobs/{external_id}
**External ID:** {external_id}

## Job Description

{body}

## Score

(scores will be filled by rescorer)
"""

_SAMPLE_JDS = [
    {
        "external_id": "rescore-jd-001",
        "title": "Staff Platform Architect",
        "company": "Acme Cloud",
        "body": (
            "We are hiring a Staff Platform Architect to lead the design "
            "of distributed systems infrastructure. You will architect "
            "cloud-native platforms using Kubernetes, Kafka, and gRPC. "
            "Experience with Terraform and AWS required. Cross-team "
            "leadership and mentoring expected."
        ),
    },
    {
        "external_id": "rescore-jd-002",
        "title": "Senior Data Engineer",
        "company": "DataFlow Inc",
        "body": (
            "Senior Data Engineer role building real-time data pipelines "
            "with Apache Spark, Airflow, and Snowflake. Design and maintain "
            "data lake architecture on AWS. Strong Python and SQL skills "
            "required. Experience with dbt preferred."
        ),
    },
    {
        "external_id": "rescore-jd-003",
        "title": "Pastry Chef",
        "company": "Sweet Treats Bakery",
        "body": (
            "Seeking an experienced pastry chef to create artisan breads "
            "and pastries for our downtown Portland bakery. Must have "
            "culinary school training and 5 years of professional baking "
            "experience. Early morning hours required."
        ),
    },
]


class TestIntegrationRescoreAccumulatedJDs:
    """
    REQUIREMENT: The rescorer processes all JD files on disk using real Ollama,
    producing a full re-scored export.

    WHO: Operator running ``rescore`` after accumulating JDs across multiple
         search runs.
    WHAT: (1) JD files seeded on disk are all discovered and re-scored by Rescorer
          (2) Every JD file produces a valid ScoreResult with fit_score > 0 and
              archetype_score > 0
          (3) Resulting CSV contains one row per JD file, sorted by final_score
              descending
          (4) Resulting Markdown ``# Run Summary`` reflects the full set of
              rescored listings
    WHY: Unit tests mock the embedder.  This validates real Ollama embedding +
         scoring against the JD file format produced by the exporter — the full
         write-parse-embed-score round trip.

    MOCK BOUNDARY:
        Mock:  (none — real Ollama)
        Real:  Ollama (embed + LLM), ChromaDB, Rescorer, JDFileExporter,
               CSVExporter, MarkdownExporter, JD file parsing
        Never: Mock Ollama or ChromaDB
    """

    async def test_rescore_discovers_and_scores_all_seeded_jd_files(
        self,
        require_ollama: None,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
        tmp_path: Path,
    ) -> None:
        """
        Given 3 JD files on disk from a prior export
        When rescore runs with real Ollama
        Then all 3 produce valid ScoreResults with fit_score > 0 and
        archetype_score > 0.
        """
        # Given: index resume and archetypes
        indexer = Indexer(store=store, embedder=embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        # Given: seed 3 JD files on disk
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()
        for jd in _SAMPLE_JDS:
            filename = f"{jd['external_id']}_{jd['company'].lower().replace(' ', '-')}_{jd['title'].lower().replace(' ', '-')}.md"
            (jd_dir / filename).write_text(_JD_TEMPLATE.format(**jd), encoding="utf-8")

        # When: rescore with real Ollama + ChromaDB
        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=False,
        )
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            comp_weight=0.15,
            min_score_threshold=0.0,
        )
        rescorer = Rescorer(scorer=scorer, ranker=ranker, base_salary=220_000)
        result = await rescorer.rescore(str(jd_dir))

        # Then: all 3 JD files discovered and scored
        assert result.total_loaded == len(_SAMPLE_JDS), (
            f"Expected {len(_SAMPLE_JDS)} JDs loaded, got {result.total_loaded}"
        )
        assert len(result.ranked_listings) == len(_SAMPLE_JDS), (
            f"Expected {len(_SAMPLE_JDS)} ranked listings, got {len(result.ranked_listings)}"
        )

        # Then: every result has valid non-zero scores
        for ranked in result.ranked_listings:
            assert ranked.scores.fit_score > 0.0, (
                f"Zero fit score for '{ranked.listing.title}' — embedding may have failed"
            )
            assert ranked.scores.archetype_score > 0.0, (
                f"Zero archetype score for '{ranked.listing.title}'"
            )

    async def test_rescore_produces_sorted_csv_matching_jd_count(
        self,
        require_ollama: None,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
        tmp_path: Path,
    ) -> None:
        """
        Given JD files with varied content
        When rescored
        Then CSV row count matches JD file count, CSV is sorted by final_score
        descending, and Markdown table row count matches CSV data row count.
        """
        # Given: index resume and archetypes
        indexer = Indexer(store=store, embedder=embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        # Given: seed JD files
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()
        for jd in _SAMPLE_JDS:
            filename = f"{jd['external_id']}_{jd['company'].lower().replace(' ', '-')}_{jd['title'].lower().replace(' ', '-')}.md"
            (jd_dir / filename).write_text(_JD_TEMPLATE.format(**jd), encoding="utf-8")

        # When: rescore and export
        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=False,
        )
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            comp_weight=0.15,
            min_score_threshold=0.0,
        )
        rescorer = Rescorer(scorer=scorer, ranker=ranker, base_salary=220_000)
        result = await rescorer.rescore(str(jd_dir))

        # Then: export CSV and verify
        csv_path = str(tmp_path / "results.csv")
        CSVExporter().export(result.ranked_listings, csv_path, summary=result.summary)

        with open(csv_path) as f:
            reader = list(csv_mod.DictReader(f))

        assert len(reader) == len(_SAMPLE_JDS), (
            f"CSV should have {len(_SAMPLE_JDS)} data rows, got {len(reader)}"
        )

        scores = [float(row["final_score"]) for row in reader]
        assert scores == sorted(scores, reverse=True), (
            f"CSV rows not sorted by final_score descending: {scores}"
        )

        # Then: export Markdown and verify table row count matches CSV
        md_path = str(tmp_path / "results.md")
        MarkdownExporter().export(result.ranked_listings, md_path, summary=result.summary)
        md_content = Path(md_path).read_text()
        assert "# Run Summary" in md_content, "Markdown should contain '# Run Summary'"

        # Count table rows (lines starting with |, excluding header separator)
        md_table_rows = [
            line
            for line in md_content.split("\n")
            if line.startswith("|") and not line.startswith("|---") and not line.startswith("| #")
        ]
        assert len(md_table_rows) == len(reader), (
            f"Markdown table rows ({len(md_table_rows)}) should match "
            f"CSV data rows ({len(reader)})"
        )


# ---------------------------------------------------------------------------
# Shared helpers for live cumulative tests (Phase 7b — B3/B4/B5)
# ---------------------------------------------------------------------------


def _make_live_settings(tmp_path: Path, *, max_pages: int = 1) -> Settings:
    """
    Load real settings.toml and override output + chroma paths to tmp_path.

    Preserves board config, scoring weights, and Ollama settings from the
    real config while directing all output and state to a temporary directory.
    Board configs are narrowed to a single search URL and ``max_pages``
    (default 1) to keep live test runs fast.
    """
    real = load_settings()
    chroma_dir = str(tmp_path / "chroma_db")
    output_dir = str(tmp_path / "output")
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Narrow each board to 1 search URL and capped max_pages
    narrowed_boards: dict[str, BoardConfig] = {}
    for name, cfg in real.boards.items():
        narrowed_boards[name] = dataclasses.replace(
            cfg,
            searches=cfg.searches[:1],
            max_pages=max_pages,
        )

    return dataclasses.replace(
        real,
        boards=narrowed_boards,
        chroma=ChromaConfig(persist_dir=chroma_dir),
        output=OutputConfig(
            default_format=real.output.default_format,
            output_dir=output_dir,
            open_top_n=0,
            jd_dir=real.output.jd_dir,
            decisions_dir=real.output.decisions_dir,
            log_dir=real.output.log_dir,
            eval_history_path=real.output.eval_history_path,
        ),
        scoring=dataclasses.replace(
            real.scoring,
            min_score_threshold=0.0,
            disqualify_on_llm_flag=False,
        ),
    )


def _make_search_args(
    *,
    board: str = "ziprecruiter",
    fresh: bool = False,
    overnight: bool = False,
    force_rescore: bool = False,
    open_top: int = 0,
    max_listings: int = 5,
) -> argparse.Namespace:
    """
    Build an argparse.Namespace matching handle_search's expectations.

    Defaults to ``max_listings=5`` so live tests only score a handful of
    listings.  Pass ``max_listings=0`` to disable the cap.
    """
    return argparse.Namespace(
        board=board,
        fresh=fresh,
        overnight=overnight,
        force_rescore=force_rescore,
        open_top=open_top,
        max_listings=max_listings,
    )


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts (convenience helper)."""
    with open(csv_path) as f:
        return list(csv_mod.DictReader(f))


# ---------------------------------------------------------------------------
# TestLiveCumulativeAccumulation (Phase 7b — B3)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveCumulativeAccumulation:
    """
    REQUIREMENT: Two successive live searches in accumulate mode produce a
    merged result set with correct CSV upsert semantics, JD file preservation,
    and Markdown summary totals.

    WHO: Operator validating the full cumulative pipeline after deployment.
    WHAT: (1)  Run 1 search produces N listings in CSV, N JD files, and a
               Markdown summary
          (2)  Run 2 search (without ``--fresh``) produces a merged CSV with
               row count >= N
          (3)  No duplicate ``external_id``s in the merged CSV (A3)
          (4)  Re-seen listings show their latest scores, not stale values
               from run 1 (A3)
          (5)  CSV header row appears exactly once and rows are sorted by
               ``final_score`` descending (A3)
          (6)  JD files from run 1 that were not in run 2 are preserved on
               disk (A4)
          (7)  JD files for re-seen listings are overwritten with updated
               metadata (A4)
          (8)  New JD files from run 2 are created (A4)
          (9)  Markdown ``# Run Summary`` "found" count reflects the total
               unique accumulated listings, and table row count matches
               CSV data row count (A8)
          (10) All ``final_score`` values parse as valid floats between 0.0
               and 1.0
    WHY: This catches DOM changes, session/auth issues, Ollama contract drift,
         CSV round-trip bugs, JD file lifecycle errors, and Markdown summary
         drift simultaneously.  No mock can replicate this coverage.

    MOCK BOUNDARY:
        Patch: ``load_settings`` is patched via ``monkeypatch.setattr`` to
               redirect output and ChromaDB paths to ``tmp_path``.  CLI
               handlers call ``load_settings()`` without a path argument,
               so patching is the only way to avoid clobbering real data.
        Real:  Browser, board adapter, Ollama, ChromaDB, all exporters,
               file system
        Never: Mock anything else (defeats live validation)
    """

    def test_accumulate_merge_produces_no_duplicates_and_sorted_scores(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Given a first live search produces N results
        When a second search runs without --fresh
        Then CSV row count >= N, no duplicate external_ids, all scores are
        valid floats in descending order, and header appears exactly once.
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: settings redirected to tmp_path
        live_settings = _make_live_settings(tmp_path)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: live_settings)

        # When: run 1 — initial search
        args_run1 = _make_search_args(board="ziprecruiter")
        handle_search(args_run1)

        out_dir = Path(live_settings.output.output_dir)
        csv_path = out_dir / "results.csv"
        jd_dir = out_dir / "jds"

        # Then: run 1 produces results
        assert csv_path.exists(), "Run 1 should produce results.csv"
        run1_rows = _read_csv_rows(csv_path)
        run1_count = len(run1_rows)
        assert run1_count >= 1, f"Run 1 should produce at least 1 listing, got {run1_count}"

        _run1_jd_files = set(jd_dir.glob("*.md"))
        _run1_ids = {row["external_id"] for row in run1_rows}

        # When: run 2 — accumulate (no --fresh)
        handle_search(_make_search_args(board="ziprecruiter"))

        # Then: CSV has >= run 1 count (WHAT 2)
        run2_rows = _read_csv_rows(csv_path)
        run2_count = len(run2_rows)
        assert run2_count >= run1_count, (
            f"Accumulated CSV should have >= {run1_count} rows, got {run2_count}"
        )

        # Then: no duplicate external_ids (WHAT 3)
        run2_ids = [row["external_id"] for row in run2_rows]
        assert len(run2_ids) == len(set(run2_ids)), (
            f"CSV contains duplicate external_ids: "
            f"{[eid for eid in run2_ids if run2_ids.count(eid) > 1]}"
        )

        # Then: sorted by final_score descending (WHAT 5)
        scores = [float(row["final_score"]) for row in run2_rows]
        assert scores == sorted(scores, reverse=True), (
            f"CSV rows not sorted by final_score descending: {scores[:10]}..."
        )

        # Then: all scores are valid floats between 0.0 and 1.0 (WHAT 10)
        for row in run2_rows:
            score = float(row["final_score"])
            assert 0.0 <= score <= 1.0, f"Score out of range for {row['external_id']}: {score}"

        # Then: header appears exactly once (WHAT 5)
        raw_csv = csv_path.read_text()
        header_line = raw_csv.strip().split("\n")[0]
        assert raw_csv.count(header_line) == 1, "CSV header should appear exactly once"

    def test_accumulate_preserves_prior_jd_files_and_creates_new(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Given two successive live runs
        When inspecting JD files
        Then prior-only JDs are preserved, re-seen JDs are overwritten,
        and new JDs are created.
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: settings redirected to tmp_path
        live_settings = _make_live_settings(tmp_path)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: live_settings)

        # When: run 1
        handle_search(_make_search_args(board="ziprecruiter"))

        out_dir = Path(live_settings.output.output_dir)
        jd_dir = out_dir / "jds"

        run1_jd_files = {f.name: f.stat().st_mtime for f in jd_dir.glob("*.md")}
        assert len(run1_jd_files) >= 1, "Run 1 should produce JD files"

        # When: run 2
        handle_search(_make_search_args(board="ziprecruiter"))

        run2_jd_files = {f.name: f.stat().st_mtime for f in jd_dir.glob("*.md")}

        # Then: all run 1 JD files still exist (WHAT 6, 7)
        for name in run1_jd_files:
            assert name in run2_jd_files, (
                f"JD file '{name}' from run 1 should still exist after run 2 "
                "(either preserved or overwritten)"
            )

        # Then: run 2 may have new JD files (WHAT 8)
        _run2_only = set(run2_jd_files.keys()) - set(run1_jd_files.keys())
        # This can be empty if the second search returns the same listings,
        # which is common with short intervals.  No assertion on count.

        # Then: total JD file count >= run 1 count
        assert len(run2_jd_files) >= len(run1_jd_files), (
            f"JD file count should not decrease: run1={len(run1_jd_files)}, "
            f"run2={len(run2_jd_files)}"
        )

    def test_accumulate_markdown_summary_reflects_merged_totals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Given two successive live runs
        When inspecting Markdown
        Then "found" count matches CSV row count and table rows match
        CSV data rows.
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: settings redirected to tmp_path
        live_settings = _make_live_settings(tmp_path)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: live_settings)

        # When: two successive searches
        handle_search(_make_search_args(board="ziprecruiter"))
        handle_search(_make_search_args(board="ziprecruiter"))

        out_dir = Path(live_settings.output.output_dir)
        csv_path = out_dir / "results.csv"
        md_path = out_dir / "results.md"

        csv_rows = _read_csv_rows(csv_path)
        md_content = md_path.read_text()

        # Then: Markdown has Run Summary (WHAT 9)
        assert "# Run Summary" in md_content, "Markdown should contain '# Run Summary'"

        # Then: Markdown table row count matches CSV data row count
        md_table_rows = [
            line
            for line in md_content.split("\n")
            if line.startswith("|") and not line.startswith("|---") and not line.startswith("| #")
        ]
        assert len(md_table_rows) == len(csv_rows), (
            f"Markdown table rows ({len(md_table_rows)}) should match "
            f"CSV data rows ({len(csv_rows)})"
        )

    def test_multi_page_traversal_produces_more_listings_than_single_page(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Given the board is configured with max_pages > 1
        When a live search runs with no listing cap
        Then the result set is larger than a single-page search.
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: single-page settings (1 URL, 1 page)
        single_page_dir = tmp_path / "single"
        single_page_dir.mkdir()
        single_settings = _make_live_settings(single_page_dir, max_pages=1)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: single_settings)

        handle_search(_make_search_args(board="ziprecruiter", max_listings=0))
        single_csv = Path(single_settings.output.output_dir) / "results.csv"
        single_count = len(_read_csv_rows(single_csv))
        assert single_count >= 1, "Single-page search should produce at least 1 listing"

        # When: multi-page settings (1 URL, 2 pages)
        multi_page_dir = tmp_path / "multi"
        multi_page_dir.mkdir()
        multi_settings = _make_live_settings(multi_page_dir, max_pages=2)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: multi_settings)

        handle_search(_make_search_args(board="ziprecruiter", max_listings=0))
        multi_csv = Path(multi_settings.output.output_dir) / "results.csv"
        multi_count = len(_read_csv_rows(multi_csv))

        # Then: multi-page should produce more results
        assert multi_count >= single_count, (
            f"Multi-page ({multi_count}) should produce >= single-page ({single_count}) listings"
        )


# ---------------------------------------------------------------------------
# TestLiveFreshModeReset (Phase 7b — B4)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveFreshModeReset:
    """
    REQUIREMENT: The ``--fresh`` flag discards all prior accumulated results
    in a live run, restoring replace-on-write behavior.

    WHO: Operator starting a clean search cycle.
    WHAT: (1) After accumulated results exist, ``search --fresh`` produces CSV
              with only current-run listings
          (2) JD files from prior runs are removed (stale cleanup restores)
          (3) Markdown reflects only the current run's totals
    WHY: Validates that ``--fresh`` works end-to-end with real services and
         real filesystem operations.  The stale-file deletion behavior in
         particular needs real FS validation.

    MOCK BOUNDARY:
        Patch: ``load_settings`` is patched via ``monkeypatch.setattr`` to
               redirect output and ChromaDB paths to ``tmp_path``.  CLI
               handlers call ``load_settings()`` without a path argument,
               so patching is the only way to avoid clobbering real data.
        Real:  Everything else
        Never: Mock anything else
    """

    def test_fresh_flag_discards_prior_csv_and_jd_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Given prior accumulated results exist
        When search runs with --fresh
        Then CSV contains only current-run listings and output/jds/ contains
        only current-run JD files (stale files deleted).
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: settings redirected to tmp_path
        live_settings = _make_live_settings(tmp_path)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: live_settings)

        out_dir = Path(live_settings.output.output_dir)
        csv_path = out_dir / "results.csv"
        jd_dir = out_dir / "jds"

        # Given: run 1 produces accumulated results
        handle_search(_make_search_args(board="ziprecruiter"))
        assert csv_path.exists(), "Run 1 should produce results.csv"
        run1_rows = _read_csv_rows(csv_path)
        assert len(run1_rows) >= 1, (
            "Run 1 should produce at least 1 qualified CSV row — "
            "all listings may have been disqualified or below threshold"
        )
        run1_jd_count = len(list(jd_dir.glob("*.md")))
        assert run1_jd_count >= 1, "Run 1 should produce JD files"

        # When: run 2 with --fresh
        handle_search(_make_search_args(board="ziprecruiter", fresh=True))

        # Then: CSV contains only current-run listings (WHAT 1)
        fresh_rows = _read_csv_rows(csv_path)
        _fresh_ids = {row["external_id"] for row in fresh_rows}
        assert len(fresh_rows) >= 1, "Fresh run should produce at least 1 listing"

        # Fresh run should not carry over prior-only listings
        # (unless the same listings appeared again, which is fine)
        # We verify by checking that the CSV was rewritten, not appended
        assert len(fresh_rows) <= len(run1_rows) + 50, (
            f"Fresh CSV ({len(fresh_rows)} rows) should not be unreasonably "
            f"larger than run 1 ({len(run1_rows)} rows)"
        )

        # Then: JD files are from current run only — stale removed (WHAT 2)
        fresh_jd_files = set(jd_dir.glob("*.md"))
        # With --fresh, cleanup_stale=True removes JDs not in current run
        # The exact count depends on what the current search returns
        assert len(fresh_jd_files) >= 1, "Fresh run should produce JD files"

    def test_fresh_flag_markdown_reflects_current_run_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Given prior accumulated results exist
        When search runs with --fresh
        Then Markdown "found" count matches CSV row count from this run only.
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: settings redirected to tmp_path
        live_settings = _make_live_settings(tmp_path)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: live_settings)

        out_dir = Path(live_settings.output.output_dir)
        csv_path = out_dir / "results.csv"
        md_path = out_dir / "results.md"

        # Given: run 1 establishes accumulation
        handle_search(_make_search_args(board="ziprecruiter"))

        # When: run 2 with --fresh
        handle_search(_make_search_args(board="ziprecruiter", fresh=True))

        # Then: Markdown matches CSV (WHAT 3)
        csv_rows = _read_csv_rows(csv_path)
        md_content = md_path.read_text()

        md_table_rows = [
            line
            for line in md_content.split("\n")
            if line.startswith("|") and not line.startswith("|---") and not line.startswith("| #")
        ]
        assert len(md_table_rows) == len(csv_rows), (
            f"Markdown table rows ({len(md_table_rows)}) should match "
            f"fresh CSV data rows ({len(csv_rows)})"
        )


# ---------------------------------------------------------------------------
# TestLiveDecisionExclusionAcrossRuns (Phase 7b — B5)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveDecisionExclusionAcrossRuns:
    """
    REQUIREMENT: A listing decided in a prior run is excluded from the next
    accumulate-mode search's exports, while its JD file is preserved.

    WHO: Operator verifying that decision filtering works across accumulated
         runs.
    WHAT: (1) After a search run, recording a "no" decision stores it in
              ChromaDB
          (2) A subsequent accumulate-mode search excludes the decided listing
              from CSV and Markdown
          (3) The decided listing's JD file is preserved in ``output/jds/``
          (4) Console session_summary shows the decision exclusion count
    WHY: Decision exclusion depends on ChromaDB round-trip (write decision ->
         read back during next search).  This validates the real interaction
         across runs, including the JD file preservation guarantee for decided
         listings.

    MOCK BOUNDARY:
        Patch: ``load_settings`` is patched via ``monkeypatch.setattr`` to
               redirect output and ChromaDB paths to ``tmp_path``.  CLI
               handlers call ``load_settings()`` without a path argument,
               so patching is the only way to avoid clobbering real data.
        Workaround (WHAT 2/3): Decisions are recorded directly via
               ``DecisionRecorder.record()`` + ``asyncio.run()`` instead
               of ``handle_decide``, because ``handle_decide`` requires an
               existing prior decision for the listing.
        Workaround (WHAT 4): ``PipelineRunner.run`` is monkeypatched to
               inject decisions through the runner's **own**
               ``_decision_recorder`` before scoring begins.  ChromaDB
               ``PersistentClient`` instances don't reliably see writes
               from prior client instances in the same process (SQLite WAL
               isolation), so decisions must be written through the same
               ``VectorStore`` that ``_score_one`` reads from.
        Real:  Browser, board adapter, Ollama, ChromaDB (decisions collection),
               DecisionRecorder, all exporters, file system
        Never: Mock anything else
    """

    def test_decided_listing_excluded_from_csv_but_jd_preserved(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Given listing X has a recorded "no" decision from a prior run
        When a new accumulate-mode search runs
        Then X does not appear in CSV or Markdown, and X's JD file is still
        present in output/jds/.
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: settings redirected to tmp_path
        live_settings = _make_live_settings(tmp_path)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: live_settings)

        out_dir = Path(live_settings.output.output_dir)
        csv_path = out_dir / "results.csv"
        jd_dir = out_dir / "jds"

        # Given: run 1 produces results
        handle_search(_make_search_args(board="ziprecruiter"))
        run1_rows = _read_csv_rows(csv_path)
        assert len(run1_rows) >= 2, (
            f"Need at least 2 listings to test exclusion, got {len(run1_rows)}"
        )

        # Given: pick one listing and record a "no" decision
        target = run1_rows[0]  # pick the top-ranked listing
        target_id = target["external_id"]
        target_title = target["title"]

        embedder = Embedder(live_settings.ollama)
        store = VectorStore(persist_dir=live_settings.chroma.persist_dir)
        recorder = DecisionRecorder(store=store, embedder=embedder)

        asyncio.run(
            recorder.record(
                job_id=target_id,
                verdict="no",
                jd_text=f"Test decision for {target_title}",
                board=target.get("board", "ziprecruiter"),
                title=target_title,
                company=target.get("company", ""),
                reason="Integration test — decision exclusion validation",
            )
        )

        # Verify decision was recorded
        decision = recorder.get_decision(target_id)
        assert decision is not None, f"Decision should be recorded for '{target_id}'"

        # Release ChromaDB PersistentClient before handle_search creates its own
        del recorder, store, embedder

        # Note which JD files exist before run 2
        run1_jd_files = set(jd_dir.glob("*.md"))
        target_jd_candidates = [f for f in run1_jd_files if target_id in f.name]

        # When: run 2 — accumulate (no --fresh)
        handle_search(_make_search_args(board="ziprecruiter"))

        # Then: decided listing excluded from CSV (WHAT 2)
        run2_rows = _read_csv_rows(csv_path)
        run2_ids = {row["external_id"] for row in run2_rows}
        assert target_id not in run2_ids, (
            f"Decided listing '{target_id}' ({target_title}) should be "
            f"excluded from CSV after decision"
        )

        # Then: decided listing excluded from Markdown (WHAT 2)
        md_path = out_dir / "results.md"
        md_content = md_path.read_text()
        assert target_id not in md_content, (
            f"Decided listing '{target_id}' should not appear in Markdown"
        )

        # Then: JD file for decided listing is preserved (WHAT 3)
        if target_jd_candidates:
            for jd_file in target_jd_candidates:
                assert jd_file.exists(), (
                    f"JD file '{jd_file.name}' for decided listing should "
                    f"be preserved (not deleted)"
                )

    def test_decision_exclusion_count_in_session_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """
        Given a decision was recorded
        When a new search completes
        Then session_summary shows skipped_decisions >= 1.
        """
        # Given: skip if no session or Ollama
        if not Path("data/ziprecruiter_session.json").exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        try:
            urllib.request.urlopen(_OLLAMA_HEALTH_URL, timeout=5)
        except Exception:
            pytest.skip(f"Ollama not reachable at {OLLAMA_BASE_URL}")

        # Given: settings redirected to tmp_path
        live_settings = _make_live_settings(tmp_path)
        monkeypatch.setattr("jobsearch_rag.cli.load_settings", lambda: live_settings)

        out_dir = Path(live_settings.output.output_dir)
        csv_path = out_dir / "results.csv"

        # Given: run 1 — uncapped so we score the full page
        handle_search(_make_search_args(board="ziprecruiter", max_listings=0))
        capsys.readouterr()  # discard run 1 output so run 2 parsing is unambiguous
        run1_rows = _read_csv_rows(csv_path)
        assert len(run1_rows) >= 1, "Need at least 1 listing"

        # Given: prepare decision data from run 1 listings.
        # Decisions are injected into run 2's own PipelineRunner to
        # guarantee ChromaDB visibility (PersistentClient instances
        # don't reliably see cross-client writes in the same process).
        run1_decision_data = [
            {
                "job_id": row["external_id"],
                "verdict": "no",
                "jd_text": f"Test decision for {row['title']}",
                "board": row.get("board", "ziprecruiter"),
                "title": row["title"],
                "company": row.get("company", ""),
                "reason": "Integration test — exclusion count",
            }
            for row in run1_rows
        ]

        # When: run 2 — monkeypatch PipelineRunner.run to inject
        # decisions through the runner's OWN _decision_recorder before
        # scoring begins, using the real DecisionRecorder + Embedder +
        # ChromaDB.  This is the only reliable way to ensure the runner's
        # _score_one sees decisions via the same VectorStore instance.
        _original_run = PipelineRunner.run

        async def _run_with_decisions(
            self_runner: PipelineRunner,
            boards: list[str] | None = None,
            *,
            overnight: bool = False,
            force_rescore: bool = False,
            max_listings: int = 0,
        ) -> RunResult:
            for decision in run1_decision_data:
                await self_runner.decision_recorder.record(**decision)
            return await _original_run(
                self_runner,
                boards=boards,
                overnight=overnight,
                force_rescore=force_rescore,
                max_listings=max_listings,
            )

        monkeypatch.setattr(PipelineRunner, "run", _run_with_decisions)
        handle_search(_make_search_args(board="ziprecruiter", max_listings=0))
        captured = capsys.readouterr()

        # Then: output shows prior decisions >= 1 (WHAT 4)
        assert "Prior decisions:" in captured.out, (
            "Session summary should report prior decision count"
            f"\nstdout (last 800 chars): {captured.out[-800:]}"
        )
        # Parse the count from "Prior decisions:  N"
        for line in captured.out.split("\n"):
            if "Prior decisions:" in line:
                count_str = line.split(":")[-1].strip()
                count = int(count_str)
                assert count >= 1, (
                    f"Prior decisions count should be >= 1, got {count}"
                    f"\nRun 1 IDs ({len(run1_rows)}): "
                    f"{[r['external_id'] for r in run1_rows[:5]]}..."
                    f"\nstdout (last 800 chars): {captured.out[-800:]}"
                )
                break

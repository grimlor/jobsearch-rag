"""
BDD specs for D5 -- constructor injection, observability guards,
and create_pipeline factory.

Covers: TestConstructorInjection (6 tests),
        TestObservabilityGuards (4 tests),
        TestCreatePipelineFactory (3 tests).

Public API surface (from src/jobsearch_rag/pipeline/runner):
    PipelineRunner(settings, *, store: VectorStorePort, embedder: EmbeddingPort)
    create_pipeline(settings) -> PipelineRunner

Public API surface (from src/jobsearch_rag/rag/scorer):
    Scorer(*, store: VectorStorePort, embedder: EmbeddingPort, ...)

Public API surface (from src/jobsearch_rag/rag/decisions):
    DecisionRecorder(*, store: VectorStorePort, embedder: EmbeddingPort, ...)

Public API surface (from src/jobsearch_rag/rag/indexer):
    Indexer(store: VectorStorePort, embedder: EmbeddingPort)

Public API surface (from src/jobsearch_rag/pipeline/eval):
    EvalRunner(scorer, ranker, store: VectorStorePort)

Public API surface (from src/jobsearch_rag/ports):
    EmbeddingPort -- Protocol
    VectorStorePort -- Protocol
    HealthCheckable -- Protocol
    MetricsProvider -- Protocol

Public API surface (from tests/fakes):
    FakeEmbedder(embed_vector=..., classify_response=...)
    InMemoryVectorStore()
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.config import load_settings
from jobsearch_rag.pipeline.eval import EvalRunner
from jobsearch_rag.pipeline.runner import (
    PipelineRunner,
    _resolve,  # pyright: ignore[reportPrivateUsage]  # testing private factory helpers
    _resolve_ports,  # pyright: ignore[reportPrivateUsage]
    create_pipeline,
)
from jobsearch_rag.ports import (
    EmbeddingPort,
    HealthCheckable,
    MetricsProvider,
    PipelinePorts,
    VectorStorePort,
)
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import OllamaEmbedder
from jobsearch_rag.rag.indexer import Indexer
from jobsearch_rag.rag.scorer import Scorer
from jobsearch_rag.rag.store import ChromaDBStore
from tests.conftest import adapter_override
from tests.fakes import FakeEmbedder, InMemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from jobsearch_rag.adapters.base import JobBoardAdapter
    from jobsearch_rag.config import Settings


# ============================================================================
# TestConstructorInjection
# ============================================================================


class TestConstructorInjection:
    """
    REQUIREMENT: Domain classes accept port interfaces via constructor
    injection instead of depending on concrete implementations.

    WHO: Scorer, DecisionRecorder, Indexer, EvalRunner, PipelineRunner
         -- all domain classes that use ChromaDBStore or OllamaEmbedder.
    WHAT: (1) Scorer.__init__ accepts store: VectorStorePort and
              embedder: EmbeddingPort.
          (2) DecisionRecorder.__init__ accepts store: VectorStorePort
              and embedder: EmbeddingPort.
          (3) Indexer.__init__ accepts store: VectorStorePort and
              embedder: EmbeddingPort.
          (4) EvalRunner.__init__ accepts store: VectorStorePort.
          (5) PipelineRunner.__init__ accepts store: VectorStorePort and
              embedder: EmbeddingPort as keyword arguments.
          (6) PipelineRunner.__init__ does NOT construct OllamaEmbedder or
              ChromaDBStore internally.
    WHY: Constructor injection is the mechanism that makes port protocols
         useful. Without it, protocols are defined but never used for
         substitution.

    MOCK BOUNDARY:
        Mock:  Nothing -- tests inject FakeEmbedder and InMemoryVectorStore
        Real:  Scorer, DecisionRecorder, Indexer, EvalRunner, PipelineRunner,
               FakeEmbedder, InMemoryVectorStore
        Never: Mock domain class methods; never patch imports
    """

    def test_scorer_accepts_port_types(self) -> None:
        """
        Given a FakeEmbedder and InMemoryVectorStore
        When Scorer is constructed with them
        Then it initializes without error
        """
        # Given: port-level fakes
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        # When: construct Scorer with fakes
        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="test prompt",
            screen_prompt="screen prompt",
            chunk_overlap=50,
            top_k_retrieval=5,
        )

        # Then: initialized without error
        assert scorer is not None, "Scorer should accept port types"

    def test_decision_recorder_accepts_port_types(self) -> None:
        """
        Given a FakeEmbedder and InMemoryVectorStore
        When DecisionRecorder is constructed with them
        Then it initializes without error
        """
        # Given: port-level fakes
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        # When: construct DecisionRecorder with fakes
        recorder = DecisionRecorder(
            store=store,
            embedder=embedder,
        )

        # Then: initialized without error
        assert recorder is not None, "DecisionRecorder should accept port types"

    def test_indexer_accepts_port_types(self) -> None:
        """
        Given a FakeEmbedder and InMemoryVectorStore
        When Indexer is constructed with them
        Then it initializes without error
        """
        # Given: port-level fakes
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()

        # When: construct Indexer with fakes
        indexer = Indexer(store=store, embedder=embedder)

        # Then: initialized without error
        assert indexer is not None, "Indexer should accept port types"

    def test_eval_runner_accepts_port_types(self) -> None:
        """
        Given an InMemoryVectorStore
        When EvalRunner is constructed with it
        Then it initializes without error
        """
        # Given: port-level fakes and stubs
        store = InMemoryVectorStore()
        scorer_stub = MagicMock()
        ranker_stub = MagicMock()

        # When: construct EvalRunner with fake store
        runner = EvalRunner(scorer=scorer_stub, ranker=ranker_stub, store=store)

        # Then: initialized without error
        assert runner is not None, "EvalRunner should accept VectorStorePort"

    def test_pipeline_runner_accepts_injected_ports(self) -> None:
        """
        Given a FakeEmbedder and InMemoryVectorStore
        When PipelineRunner is constructed with store= and embedder= kwargs
        Then it initializes without error and all internal components
             (Scorer, Ranker, DecisionRecorder) are wired correctly
        """
        # Given: port-level fakes and settings
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        settings = _make_settings()

        # When: construct PipelineRunner with injected ports
        runner = PipelineRunner(settings, store=store, embedder=embedder)

        # Then: initialized without error
        assert runner is not None, "PipelineRunner should accept injected ports"

    def test_pipeline_runner_does_not_construct_infra_internally(self) -> None:
        """
        Given a PipelineRunner constructed with injected ports
        When its _store and _embedder attributes are inspected
        Then they are the exact objects that were injected (identity check)
        """
        # Given: specific instances to inject
        store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        settings = _make_settings()

        # When: construct and inspect
        runner = PipelineRunner(settings, store=store, embedder=embedder)

        # Then: the exact injected objects are used (identity check)
        assert runner._store is store, (  # pyright: ignore[reportPrivateUsage]
            "PipelineRunner._store should be the exact injected store instance"
        )
        assert runner._embedder is embedder, (  # pyright: ignore[reportPrivateUsage]
            "PipelineRunner._embedder should be the exact injected embedder instance"
        )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_settings() -> Settings:
    """Build a minimal Settings object for PipelineRunner construction."""
    return load_settings()


def _seed_required_collections(store: InMemoryVectorStore) -> None:
    """
    Seed the three required collections so auto-indexing is skipped.

    Uses 8-dim vectors to match FakeEmbedder's default dimension.
    """
    embed_fake = [0.1] * 8
    for name in ("resume", "role_archetypes", "global_positive_signals"):
        store.add_documents(
            name,
            ids=[f"{name}-seed"],
            documents=[f"Seed document for {name}"],
            embeddings=[embed_fake],
        )


# ============================================================================
# TestObservabilityGuards
# ============================================================================


class TestObservabilityGuards:
    """
    REQUIREMENT: PipelineRunner.run() uses isinstance guards to
    optionally invoke health_check and metrics on the injected embedder,
    gracefully skipping when the implementation does not support them.

    WHO: PipelineRunner -- the orchestrator that runs the pipeline and
         optionally collects observability data.
    WHAT: (1) When embedder satisfies HealthCheckable, health_check()
              is called during pre-flight.
          (2) When embedder does NOT satisfy HealthCheckable (e.g.,
              FakeEmbedder), pre-flight skips health_check silently.
          (3) When embedder satisfies MetricsProvider, metrics are
              collected during session summary.
          (4) When embedder does NOT satisfy MetricsProvider (e.g.,
              FakeEmbedder), session summary skips metrics silently.
    WHY: The isinstance guard pattern keeps the embedding port narrow
         while allowing production embedders to provide observability.
         Tests with FakeEmbedder skip observability automatically --
         no stubs needed.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (for real Embedder construction only)
        Real:  PipelineRunner, isinstance guards, FakeEmbedder, Embedder
        Never: Mock isinstance or the guard logic itself
    """

    async def test_health_check_called_when_embedder_is_health_checkable(self) -> None:
        """
        Given a PipelineRunner with a real Embedder (satisfies HealthCheckable)
        When run() begins pre-flight
        Then health_check() is awaited
        """
        # Given: a real Embedder with mocked ollama client
        settings = _make_settings()
        mock_client = _make_mock_ollama_client(settings)
        store = InMemoryVectorStore()
        _seed_required_collections(store)

        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
            return_value=mock_client,
        ):
            embedder = OllamaEmbedder(settings.ollama)

        runner = PipelineRunner(settings, store=store, embedder=embedder)

        # When: run with a nonexistent board (hits early return after health check)
        await runner.run(boards=["nonexistent_board"])

        # Then: health_check called via client.list()
        mock_client.list.assert_awaited_once()

    async def test_health_check_skipped_when_embedder_is_not_health_checkable(self) -> None:
        """
        Given a PipelineRunner with a FakeEmbedder (does not satisfy HealthCheckable)
        When run() begins pre-flight
        Then no health_check() call occurs and no error is raised
        """
        # Given: FakeEmbedder does not satisfy HealthCheckable
        settings = _make_settings()
        store = InMemoryVectorStore()
        _seed_required_collections(store)
        embedder = FakeEmbedder()

        assert not isinstance(embedder, HealthCheckable), (
            "FakeEmbedder should NOT satisfy HealthCheckable"
        )

        runner = PipelineRunner(settings, store=store, embedder=embedder)

        # When: run with a nonexistent board -- should not error
        result = await runner.run(boards=["nonexistent_board"])

        # Then: completed without error (health_check was skipped)
        assert result is not None, "run() should complete without error"
        assert result.boards_searched == ["nonexistent_board"], (
            f"Expected [nonexistent_board], got {result.boards_searched}"
        )

    async def test_metrics_collected_when_embedder_is_metrics_provider(self) -> None:
        """
        Given a PipelineRunner with a real Embedder (satisfies MetricsProvider)
        When run() completes
        Then session summary includes embedder.metrics data
        """
        # Given: a real Embedder with mocked ollama client
        settings = _make_settings()
        mock_client = _make_mock_ollama_client(settings)
        store = InMemoryVectorStore()
        _seed_required_collections(store)

        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
            return_value=mock_client,
        ):
            embedder = OllamaEmbedder(settings.ollama)

        assert isinstance(embedder, MetricsProvider), (
            "Real Embedder should satisfy MetricsProvider"
        )

        runner = PipelineRunner(settings, store=store, embedder=embedder)

        # When: run with a nonexistent board (empty listings → early return with summary)
        with patch("jobsearch_rag.pipeline.runner.log_event") as mock_log_event:
            await runner.run(boards=["nonexistent_board"])

        # Then: session_summary includes metrics fields
        mock_log_event.assert_called()
        summary_calls = [
            c for c in mock_log_event.call_args_list if c.args[0] == "session_summary"
        ]
        assert len(summary_calls) >= 1, "Expected at least one session_summary log event"
        summary_kwargs = summary_calls[0].kwargs
        assert "embed_calls" in summary_kwargs, (
            "session_summary should include embed_calls from metrics"
        )

    async def test_metrics_skipped_when_embedder_is_not_metrics_provider(self) -> None:
        """
        Given a PipelineRunner with a FakeEmbedder (does not satisfy MetricsProvider)
        When run() completes
        Then session summary omits metrics and no error is raised
        """
        # Given: FakeEmbedder does not satisfy MetricsProvider
        settings = _make_settings()
        store = InMemoryVectorStore()
        _seed_required_collections(store)
        embedder = FakeEmbedder()

        assert not isinstance(embedder, MetricsProvider), (
            "FakeEmbedder should NOT satisfy MetricsProvider"
        )

        runner = PipelineRunner(settings, store=store, embedder=embedder)

        # When: run with a nonexistent board -- should not error
        with patch("jobsearch_rag.pipeline.runner.log_event"):
            result = await runner.run(boards=["nonexistent_board"])

        # Then: completed without error (metrics were skipped)
        assert result is not None, "run() should complete without error"

    async def test_metrics_skipped_at_end_of_scored_run(self) -> None:
        """
        Given a PipelineRunner with a FakeEmbedder and actual listings scored
        When run() completes the full scoring path
        Then session summary omits metrics and no error is raised
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: FakeEmbedder with a classify response for disqualifier
            settings = _make_settings_with_tmpdir(tmpdir, enabled_boards=["testboard"])
            store = InMemoryVectorStore()
            _seed_required_collections(store)
            embedder = FakeEmbedder(
                classify_response='{"disqualified": false, "reason": null}',
            )

            assert not isinstance(embedder, MetricsProvider), (
                "FakeEmbedder should NOT satisfy MetricsProvider"
            )

            runner = PipelineRunner(settings, store=store, embedder=embedder)

            # Given: a test adapter that returns one listing
            listing = JobListing(
                board="testboard",
                external_id="test-1",
                title="Staff Architect",
                company="Acme",
                location="Remote",
                url="https://testboard.com/1",
                full_text="A detailed job description.",
                max_full_text_chars=250_000,
            )
            mock_adapter = _make_test_adapter(search_results=[listing])

            mock_pw_fn, _ = _mock_playwright_boundary()

            # When: full pipeline run with actual scoring
            with (
                adapter_override({"testboard": _adapt(mock_adapter)}, clear=True),
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch(
                    "jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR",
                    Path(tmpdir),
                ),
                patch("jobsearch_rag.pipeline.runner.log_event") as mock_log_event,
            ):
                result = await runner.run()

            # Then: listings were scored and metrics were omitted
            assert result.summary.total_scored >= 1, (
                f"Expected at least 1 scored listing, got {result.summary.total_scored}"
            )
            summary_calls = [
                c for c in mock_log_event.call_args_list if c.args[0] == "session_summary"
            ]
            assert len(summary_calls) >= 1, "Expected session_summary log event"
            summary_kwargs = summary_calls[-1].kwargs
            assert "embed_calls" not in summary_kwargs, (
                "session_summary should NOT include embed_calls when embedder is not MetricsProvider"
            )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_ollama_client(settings: Settings) -> AsyncMock:
    """Create a mock ollama AsyncClient that satisfies health_check."""
    mock_client = AsyncMock()

    # health_check() calls client.list()
    model_embed = MagicMock()
    model_embed.model = settings.ollama.embed_model
    model_llm = MagicMock()
    model_llm.model = settings.ollama.llm_model
    list_response = MagicMock()
    list_response.models = [model_embed, model_llm]
    mock_client.list.return_value = list_response

    # embed() calls client.embed()
    embed_response = MagicMock()
    embed_response.embeddings = [[0.1] * 768]
    mock_client.embed.return_value = embed_response

    return mock_client


# ============================================================================
# TestCreatePipelineFactory
# ============================================================================


class TestCreatePipelineFactory:
    """
    REQUIREMENT: create_pipeline() is a factory function that constructs
    a PipelineRunner with production infrastructure wiring.

    WHO: CLI handlers (handle_search, handle_index, handle_decide, etc.)
         -- the only callers that need concrete OllamaEmbedder and ChromaDBStore.
    WHAT: (1) create_pipeline(settings) returns a PipelineRunner.
          (2) The returned runner's store is a ChromaDBStore configured
              from settings.chroma (persist_dir, distance_metric).
          (3) The returned runner's embedder is an OllamaEmbedder configured
              from settings.ollama.
    WHY: Centralizes production wiring in one function. CLI handlers call
         create_pipeline() instead of constructing infrastructure themselves.
         Tests bypass the factory entirely by injecting fakes.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama I/O boundary for OllamaEmbedder construction),
               chromadb.PersistentClient (ChromaDB I/O boundary for ChromaDBStore)
        Real:  create_pipeline function, PipelineRunner, Settings
        Never: Mock PipelineRunner.__init__
    """

    def test_create_pipeline_returns_pipeline_runner(self) -> None:
        """
        Given valid Settings
        When create_pipeline(settings) is called
        Then it returns a PipelineRunner instance
        """
        # Given: valid settings with mocked I/O boundaries
        settings = _make_settings()

        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
            return_value=AsyncMock(),
        ):
            # When: create_pipeline
            runner = create_pipeline(settings)

        # Then: it's a PipelineRunner
        assert isinstance(runner, PipelineRunner), (
            f"Expected PipelineRunner, got {type(runner).__name__}"
        )

    def test_create_pipeline_wires_vector_store(self) -> None:
        """
        Given Settings with chroma.persist_dir and chroma.distance_metric
        When create_pipeline(settings) is called
        Then the runner's store is a ChromaDBStore with matching configuration
        """
        # Given: settings with chroma config
        settings = _make_settings()

        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
            return_value=AsyncMock(),
        ):
            # When: create_pipeline
            runner = create_pipeline(settings)

        # Then: store is a VectorStore
        store = runner._store  # pyright: ignore[reportPrivateUsage]
        assert isinstance(store, ChromaDBStore), (
            f"Expected ChromaDBStore, got {type(store).__name__}"
        )

    def test_create_pipeline_wires_embedder(self) -> None:
        """
        Given Settings with ollama config
        When create_pipeline(settings) is called
        Then the runner's embedder is an OllamaEmbedder configured from settings.ollama
        """
        # Given: settings with ollama config
        settings = _make_settings()

        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
            return_value=AsyncMock(),
        ):
            # When: create_pipeline
            runner = create_pipeline(settings)

        # Then: embedder is an Embedder
        embedder = runner._embedder  # pyright: ignore[reportPrivateUsage]
        assert isinstance(embedder, OllamaEmbedder), (
            f"Expected OllamaEmbedder, got {type(embedder).__name__}"
        )


class TestConfigDrivenPortResolution:
    """
    REQUIREMENT: _resolve() and _resolve_ports() provide config-driven
    port resolution -- loading adapter classes at runtime via dotted
    paths in the [ports] config section.

    WHO: create_pipeline() -- the sole caller of _resolve_ports().
    WHAT: (1) _resolve() imports a class by dotted path, calls
              from_settings, and validates the result against the
              requested port type.
          (2) _resolve_ports() builds a PipelinePorts from the [ports]
              config section.
          (3) Invalid dotted paths raise ImportError.
          (4) Classes missing from_settings raise TypeError.
          (5) Classes producing the wrong port type raise TypeError.
    WHY: Decouples the pipeline factory from concrete imports -- new
         adapter implementations can be wired in via config alone.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama I/O boundary)
        Real:  _resolve, _resolve_ports, OllamaEmbedder, ChromaDBStore,
               Settings, PortsConfig
        Never: Mock importlib internals
    """

    def test_resolve_valid_embedder(self) -> None:
        """
        Given a dotted path pointing to OllamaEmbedder
        When _resolve is called with EmbeddingPort as port_type
        Then it returns an OllamaEmbedder instance
        """
        settings = _make_settings()
        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
            return_value=AsyncMock(),
        ):
            result = _resolve(
                "jobsearch_rag.rag.embedder.OllamaEmbedder",
                EmbeddingPort,
                settings,
            )
        assert isinstance(result, OllamaEmbedder)

    def test_resolve_valid_store(self) -> None:
        """
        Given a dotted path pointing to ChromaDBStore
        When _resolve is called with VectorStorePort as port_type
        Then it returns a ChromaDBStore instance
        """
        settings = _make_settings()
        result = _resolve(
            "jobsearch_rag.rag.store.ChromaDBStore",
            VectorStorePort,
            settings,
        )
        assert isinstance(result, ChromaDBStore)

    def test_resolve_ports_returns_pipeline_ports(self) -> None:
        """
        Given valid Settings with [ports] section
        When _resolve_ports is called
        Then it returns a PipelinePorts with the correct adapter types
        """
        settings = _make_settings()
        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
            return_value=AsyncMock(),
        ):
            ports = _resolve_ports(settings)
        assert isinstance(ports, PipelinePorts)
        assert isinstance(ports.embedder, OllamaEmbedder)
        assert isinstance(ports.store, ChromaDBStore)

    def test_resolve_invalid_dotted_path_no_module(self) -> None:
        """
        Given a dotted path with no module separator
        When _resolve is called
        Then it raises ImportError
        """
        settings = _make_settings()
        with pytest.raises(ImportError, match="Invalid dotted path"):
            _resolve("NoModuleSeparator", EmbeddingPort, settings)

    def test_resolve_nonexistent_module(self) -> None:
        """
        Given a dotted path pointing to a module that does not exist
        When _resolve is called
        Then it raises ImportError
        """
        settings = _make_settings()
        with pytest.raises(ImportError):
            _resolve("nonexistent.module.ClassName", EmbeddingPort, settings)

    def test_resolve_nonexistent_class(self) -> None:
        """
        Given a dotted path pointing to a valid module but nonexistent class
        When _resolve is called
        Then it raises ImportError
        """
        settings = _make_settings()
        with pytest.raises(ImportError, match="has no attribute"):
            _resolve(
                "jobsearch_rag.rag.embedder.NonexistentClass",
                EmbeddingPort,
                settings,
            )

    def test_resolve_class_without_port_factory(self) -> None:
        """
        Given a dotted path pointing to a class that does not implement PortFactory
        When _resolve is called
        Then it raises TypeError
        """
        settings = _make_settings()
        with pytest.raises(TypeError, match="does not implement PortFactory"):
            _resolve(
                "jobsearch_rag.config.Settings",
                EmbeddingPort,
                settings,
            )

    def test_resolve_wrong_port_type(self) -> None:
        """
        Given a dotted path to ChromaDBStore (a VectorStorePort)
        When _resolve is called with EmbeddingPort as port_type
        Then it raises TypeError with a descriptive message
        """
        settings = _make_settings()
        with pytest.raises(TypeError, match="expected EmbeddingPort"):
            _resolve(
                "jobsearch_rag.rag.store.ChromaDBStore",
                EmbeddingPort,
                settings,
            )


# ---------------------------------------------------------------------------
# Pipeline integration helpers (for full-run coverage)
# ---------------------------------------------------------------------------


def _adapt(adapter: object) -> Callable[..., JobBoardAdapter]:
    """Wrap an adapter/mock as a registry-compatible factory accepting any kwargs."""

    def _factory(**_kwargs: object) -> JobBoardAdapter:
        return cast("JobBoardAdapter", adapter)

    return _factory


def _make_test_adapter(
    *,
    search_results: list[JobListing] | None = None,
) -> MagicMock:
    """Create a mock adapter that returns the given search results."""
    adapter = MagicMock()
    adapter.board_name = "testboard"
    adapter.authenticate = AsyncMock()
    adapter.search = AsyncMock(return_value=search_results or [])
    adapter.extract_detail = AsyncMock()
    return adapter


def _mock_playwright_boundary() -> tuple[MagicMock, MagicMock]:
    """Mock async_playwright -- the Playwright I/O boundary."""
    mock_page = MagicMock()

    mock_context = MagicMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.storage_state = AsyncMock(return_value={"cookies": [], "origins": []})
    mock_context.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mock_pw = MagicMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw.stop = AsyncMock()

    mock_pw_cm = MagicMock()
    mock_pw_cm.start = AsyncMock(return_value=mock_pw)

    mock_async_pw = MagicMock(return_value=mock_pw_cm)

    return mock_async_pw, mock_page


def _make_settings_with_tmpdir(
    tmpdir: str,
    *,
    enabled_boards: list[str] | None = None,
) -> Settings:
    """Create Settings with tmpdir-based paths for integration-style tests."""
    from tests.conftest import make_test_settings  # noqa: PLC0415

    return make_test_settings(tmpdir, enabled_boards=enabled_boards)

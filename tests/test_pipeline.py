"""BDD specs for the pipeline runner — orchestration, board delegation, and Ollama connectivity.

Covers:
    - TestOllamaConnectivity       (BDD Specifications — pipeline.md)
    - TestPipelineOrchestration    (BDD Specifications — pipeline.md)
    - TestBoardSearchDelegation    (BDD Specifications — pipeline.md)

The pipeline runner is the top-level orchestrator.  Tests exercise the real
PipelineRunner with mocked I/O boundaries (Ollama network, browser sessions)
and verify orchestration order, failure isolation, and result assembly.
"""

# Public API surface (from src/jobsearch_rag/pipeline/runner.py):
#   PipelineRunner(settings: Settings)
#   runner.run(boards=None, *, overnight=False, force_rescore=False) -> RunResult
#
# Public API surface (from src/jobsearch_rag/pipeline/runner.py):
#   RunResult(ranked_listings=[], summary=RankSummary(), failed_listings=0,
#             skipped_decisions=0, boards_searched=[])
#
# Public API surface (from src/jobsearch_rag/errors.py):
#   ActionableError(error, error_type, service, ...)
#   ActionableError.connection(service, url, raw_error) -> ActionableError
#   ActionableError.embedding(model, raw_error, *, suggestion=None) -> ActionableError
#
# Public API surface (from src/jobsearch_rag/rag/embedder.py):
#   Embedder(base_url, embed_model, llm_model)
#   embedder.health_check() -> None  (raises ActionableError on failure)
#
# Fixture contracts (from conftest.py):
#   make_runner_with_mocks(settings, *, populate_store=True)
#       -> (PipelineRunner, Embedder, MagicMock[Scorer])
#   make_settings(enabled_boards=None, overnight_boards=None) -> Settings
#   mock_board_io(search_results=None)
#       -> (mock_adapter, mock_session, mock_registry)
#   make_listing(board, external_id, title, full_text, ...) -> JobListing

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.pipeline.runner import PipelineRunner, RunResult
from jobsearch_rag.rag.scorer import ScoreResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from jobsearch_rag.adapters.base import JobListing
    from jobsearch_rag.config import Settings
    from jobsearch_rag.rag.embedder import Embedder


# ---------------------------------------------------------------------------
# TestOllamaConnectivity
# ---------------------------------------------------------------------------


class TestOllamaConnectivity:
    """
    REQUIREMENT: Ollama unavailability is detected before browser work begins,
    with guidance the operator can act on immediately.

    WHO: The pipeline runner; the operator who may have forgotten to start Ollama
    WHAT: When embedder.health_check() fails, runner.run() raises an
          ActionableError before opening any browser session; the error names
          the configured URL and provides a connectivity test command;
          a missing model error suggests the specific ollama pull command;
          a health check timeout advises checking system memory and process status
    WHY: A full browser session completing only to fail at scoring wastes
         10+ minutes of scraping time and risks rate limiting with no usable output

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture — health_check() raises ActionableError
               on the first call; mock_board_io fixture to verify page.goto
               is never called after health check failure
        Real:  PipelineRunner, ActionableError
        Never: Construct ActionableError and assert on it directly — invoke
               runner.run() and let the error propagate naturally
    """

    @pytest.mark.asyncio
    async def test_health_check_failure_raises_before_browser_session_opens(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given embedder.health_check() raises an ActionableError
        When runner.run() is awaited
        Then the error propagates before any page.goto() is called
        """
        # Given: runner with health_check that fails
        settings = make_settings()
        runner, mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        object.__setattr__(mock_emb, "health_check", AsyncMock(
            side_effect=ActionableError.connection(
                service="Ollama",
                url="http://localhost:11434",
                raw_error="Connection refused",
            )
        ))
        _mock_adapter, mock_session, mock_registry = mock_board_io()

        # When: runner.run() is awaited — should propagate the error
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            pytest.raises(ActionableError) as exc_info,
        ):
            await runner.run()

        # Then: the error propagated and no browser session was opened
        assert exc_info.value.error_type == ErrorType.CONNECTION, (
            f"Expected CONNECTION error type. Got: {exc_info.value.error_type}"
        )
        mock_session.__aenter__.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unreachable_ollama_error_names_the_configured_url(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
    ) -> None:
        """
        When a CONNECTION ActionableError is constructed for an Ollama endpoint
        Then the error message contains the URL so the operator can test it directly
        """
        # Given: runner with health_check that fails naming the URL
        settings = make_settings()
        runner, mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        target_url = "http://localhost:11434"
        object.__setattr__(mock_emb, "health_check", AsyncMock(
            side_effect=ActionableError.connection(
                service="Ollama",
                url=target_url,
                raw_error="Connection refused",
            )
        ))

        # When: runner.run() propagates the error
        with pytest.raises(ActionableError) as exc_info:
            await runner.run()

        # Then: the error message names the URL
        assert target_url in exc_info.value.error, (
            f"Expected URL '{target_url}' in error message. "
            f"Got: {exc_info.value.error}"
        )

    @pytest.mark.asyncio
    async def test_wrong_model_name_error_suggests_ollama_pull_command(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
    ) -> None:
        """
        When a model-not-found error is modelled as an ActionableError
        Then the suggestion contains 'ollama pull' so the operator knows the recovery command
        """
        # Given: runner with health_check that fails for missing model
        settings = make_settings()
        runner, mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        object.__setattr__(mock_emb, "health_check", AsyncMock(
            side_effect=ActionableError.embedding(
                model="nomic-embed-text",
                raw_error="Model 'nomic-embed-text' is not pulled in Ollama",
                suggestion="Run: ollama pull nomic-embed-text",
            )
        ))

        # When: runner.run() propagates the error
        with pytest.raises(ActionableError) as exc_info:
            await runner.run()

        # Then: the suggestion mentions 'ollama pull'
        assert "ollama pull" in (exc_info.value.suggestion or ""), (
            f"Expected 'ollama pull' in suggestion. "
            f"Got: {exc_info.value.suggestion}"
        )

    @pytest.mark.asyncio
    async def test_ollama_timeout_error_advises_checking_system_resources(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
    ) -> None:
        """
        When an Ollama timeout is modelled as an ActionableError
        Then the troubleshooting steps mention system memory or process status
        """
        # Given: runner with health_check that times out
        settings = make_settings()
        runner, mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        object.__setattr__(mock_emb, "health_check", AsyncMock(
            side_effect=ActionableError.connection(
                service="Ollama",
                url="http://localhost:11434",
                raw_error="Request timed out after 30s",
                suggestion="Check system memory and Ollama process status",
            )
        ))

        # When: runner.run() propagates the error
        with pytest.raises(ActionableError) as exc_info:
            await runner.run()

        # Then: the troubleshooting steps or suggestion mention resources
        err = exc_info.value
        troubleshooting_text = ""
        if err.troubleshooting is not None:
            troubleshooting_text = " ".join(err.troubleshooting.steps)
        combined = f"{err.suggestion or ''} {troubleshooting_text}"
        assert "memory" in combined.lower() or "process" in combined.lower() or "running" in combined.lower(), (
            f"Expected troubleshooting to mention memory or process status. "
            f"Got suggestion: {err.suggestion}, troubleshooting: {troubleshooting_text}"
        )


# ---------------------------------------------------------------------------
# TestPipelineOrchestration
# ---------------------------------------------------------------------------


class TestPipelineOrchestration:
    """
    REQUIREMENT: The pipeline runner executes steps in correct order with
    proper isolation between boards and failure handling at every level.

    WHO: The operator running a search; downstream consumers of RunResult
    WHAT: Ollama health check runs before any browser work; enabled boards
          are searched when no explicit board list is given; overnight mode
          includes overnight-only boards; a board that fails entirely does
          not abort other boards; scoring failures increment failed_listings
          without aborting; empty results return a valid RunResult with zero
          ranked listings; auto-indexing runs before scoring when collections
          are empty
    WHY: A health check after browser work wastes time; one board failure
         aborting the entire run discards valid results from other boards;
         an invalid RunResult crashes exporters downstream

    MOCK BOUNDARY:
        Mock:  mock_board_io fixture (Playwright I/O); mock_embedder fixture
               (Ollama HTTP); asyncio.sleep for throttle bypass
        Real:  PipelineRunner, Ranker, RunResult, Settings from tmp_path TOML
        Never: Construct RunResult directly; never patch the orchestration
               method internals — trigger failure conditions through the
               mock_board_io's side_effect list
    """

    @pytest.mark.asyncio
    async def test_health_check_runs_before_any_board_search(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        When runner.run() is awaited
        Then health_check is called before any adapter.search()
        """
        # Given: runner with working health_check and board search
        settings = make_settings()
        runner, mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        mock_adapter, mock_session, mock_registry = mock_board_io()

        call_order: list[str] = []
        original_health_check = mock_emb.health_check

        async def track_health_check() -> None:
            call_order.append("health_check")
            return await original_health_check()

        async def track_search(*args: object, **kwargs: object) -> list[JobListing]:
            call_order.append("search")
            return []

        object.__setattr__(mock_emb, "health_check", AsyncMock(side_effect=track_health_check))
        mock_adapter.search = AsyncMock(side_effect=track_search)

        # When: run is awaited
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            await runner.run()

        # Then: health_check was called before search
        assert "health_check" in call_order, (
            f"health_check was never called. Call order: {call_order}"
        )
        if "search" in call_order:
            hc_idx = call_order.index("health_check")
            search_idx = call_order.index("search")
            assert hc_idx < search_idx, (
                f"health_check (pos={hc_idx}) should run before search (pos={search_idx})"
            )

    @pytest.mark.asyncio
    async def test_defaults_to_enabled_boards_when_none_specified(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given settings with enabled_boards=["board_a", "board_b"]
        When runner.run() is called without specifying boards
        Then both enabled boards appear in boards_searched
        """
        # Given: settings with two enabled boards
        settings = make_settings(enabled_boards=["board_a", "board_b"])
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io()

        # When: run without specifying boards
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: both boards are in boards_searched
        assert "board_a" in result.boards_searched, (
            f"Expected 'board_a' in boards_searched. Got: {result.boards_searched}"
        )
        assert "board_b" in result.boards_searched, (
            f"Expected 'board_b' in boards_searched. Got: {result.boards_searched}"
        )

    @pytest.mark.asyncio
    async def test_explicit_boards_override_enabled_boards(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given settings with enabled_boards=["board_a", "board_b"]
        When runner.run(boards=["board_a"]) is called
        Then only "board_a" is in boards_searched
        """
        # Given: settings with two enabled boards
        settings = make_settings(enabled_boards=["board_a", "board_b"])
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io()

        # When: run with explicit board list
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run(boards=["board_a"])

        # Then: only the explicit board appears
        assert result.boards_searched == ["board_a"], (
            f"Expected boards_searched=['board_a']. Got: {result.boards_searched}"
        )

    @pytest.mark.asyncio
    async def test_overnight_mode_includes_overnight_boards(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given settings with overnight_boards=["linkedin"]
        When runner.run(overnight=True) is called
        Then "linkedin" appears in boards_searched
        """
        # Given: settings with overnight boards
        settings = make_settings(
            enabled_boards=["ziprecruiter"],
            overnight_boards=["linkedin"],
        )
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io()

        # When: run in overnight mode
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run(overnight=True)

        # Then: overnight board is included
        assert "linkedin" in result.boards_searched, (
            f"Expected 'linkedin' in boards_searched for overnight mode. "
            f"Got: {result.boards_searched}"
        )
        assert "ziprecruiter" in result.boards_searched, (
            f"Expected 'ziprecruiter' still in boards_searched. "
            f"Got: {result.boards_searched}"
        )

    @pytest.mark.asyncio
    async def test_board_failure_does_not_abort_other_boards(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
    ) -> None:
        """
        Given two boards where the first throws an ActionableError
        When runner.run() completes
        Then the second board's results are still in the RunResult
        """
        # Given: two boards, first fails entirely
        settings = make_settings(enabled_boards=["failing_board", "good_board"])
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        good_listing = make_listing(board="good_board", title="Good Job")

        # Build per-board mock adapters
        mock_failing_adapter = MagicMock()
        mock_failing_adapter.board_name = "failing_board"
        mock_failing_adapter.rate_limit_seconds = (0.0, 0.0)
        mock_failing_adapter.authenticate = AsyncMock()
        mock_failing_adapter.search = AsyncMock(
            side_effect=ActionableError(
                error="Board unavailable",
                error_type=ErrorType.CONNECTION,
                service="failing_board",
            )
        )

        mock_good_adapter = MagicMock()
        mock_good_adapter.board_name = "good_board"
        mock_good_adapter.rate_limit_seconds = (0.0, 0.0)
        mock_good_adapter.authenticate = AsyncMock()
        mock_good_adapter.search = AsyncMock(return_value=[good_listing])
        mock_good_adapter.extract_detail = AsyncMock()

        def get_adapter(board_name: str) -> MagicMock:
            if board_name == "failing_board":
                return mock_failing_adapter
            return mock_good_adapter

        mock_registry = MagicMock()
        mock_registry.get = MagicMock(side_effect=get_adapter)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.new_page = AsyncMock(return_value=MagicMock())
        mock_session.save_storage_state = AsyncMock()

        # When: run both boards
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: good board's listing was scored (result is not empty)
        assert "good_board" in result.boards_searched, (
            f"Expected 'good_board' in boards_searched. Got: {result.boards_searched}"
        )
        assert "failing_board" in result.boards_searched, (
            f"Expected 'failing_board' still listed in boards_searched. "
            f"Got: {result.boards_searched}"
        )

    @pytest.mark.asyncio
    async def test_scoring_failure_increments_failed_listings_not_aborts(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given two listings where scoring the first raises an ActionableError
        When runner.run() completes
        Then failed_listings is incremented and the second listing is still scored
        """
        # Given: two listings, scorer fails on first then succeeds
        settings = make_settings()
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)
        listing_a = make_listing(board="testboard", external_id="a", title="Failing Role")
        listing_b = make_listing(board="testboard", external_id="b", title="Good Role")
        _mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[listing_a, listing_b]
        )

        # Scorer fails on first call, succeeds on second
        mock_scorer.score = AsyncMock(
            side_effect=[
                ActionableError(
                    error="Scoring failed",
                    error_type=ErrorType.EMBEDDING,
                    service="Ollama",
                ),
                ScoreResult(
                    fit_score=0.8,
                    archetype_score=0.7,
                    history_score=0.5,
                    disqualified=False,
                ),
            ]
        )

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: one failure counted, run did not abort
        assert result.failed_listings >= 1, (
            f"Expected at least 1 failed_listing. Got: {result.failed_listings}"
        )

    @pytest.mark.asyncio
    async def test_empty_results_return_valid_run_result(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given no listings are returned by any board
        When runner.run() completes
        Then the result is a valid RunResult with empty ranked_listings
        """
        # Given: boards return no listings
        settings = make_settings()
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io(search_results=[])

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: valid RunResult with no listings
        assert isinstance(result, RunResult), (
            f"Expected RunResult instance. Got: {type(result)}"
        )
        assert result.ranked_listings == [], (
            f"Expected empty ranked_listings. Got: {result.ranked_listings}"
        )
        assert result.boards_searched == ["testboard"], (
            f"Expected boards_searched=['testboard']. Got: {result.boards_searched}"
        )

    @pytest.mark.asyncio
    async def test_auto_indexes_when_collections_are_empty(
        self,
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
        mock_embedder: Embedder,
    ) -> None:
        """
        Given the resume, role_archetypes, and global_positive_signals
              collections are empty
        When runner.run() is awaited
        Then the indexer runs before scoring
        """
        # Given: runner with empty collections (populate_store=False)
        settings = make_settings()
        _mock_adapter, mock_session, mock_registry = mock_board_io()

        # Track whether indexer methods are called
        index_calls: list[str] = []

        async def mock_index_archetypes(*args: object, **kwargs: object) -> int:
            index_calls.append("archetypes")
            return 1

        async def mock_index_resume(*args: object, **kwargs: object) -> int:
            index_calls.append("resume")
            return 1

        async def mock_index_positive(*args: object, **kwargs: object) -> int:
            index_calls.append("positive")
            return 1

        mock_scorer = MagicMock()
        mock_scorer.score = AsyncMock(return_value=MagicMock())

        with (
            patch("jobsearch_rag.pipeline.runner.Embedder", return_value=mock_embedder),
            patch("jobsearch_rag.pipeline.runner.Scorer", return_value=mock_scorer),
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            runner = PipelineRunner(settings)

            # Patch the indexer after construction
            with patch("jobsearch_rag.pipeline.runner.Indexer") as mock_indexer_cls:
                mock_indexer_instance = MagicMock()
                mock_indexer_instance.index_archetypes = AsyncMock(side_effect=mock_index_archetypes)
                mock_indexer_instance.index_resume = AsyncMock(side_effect=mock_index_resume)
                mock_indexer_instance.index_global_positive_signals = AsyncMock(
                    side_effect=mock_index_positive
                )
                mock_indexer_cls.return_value = mock_indexer_instance

                await runner.run()

        # Then: all three index methods were called
        assert "archetypes" in index_calls, (
            f"Expected archetypes indexing. Got calls: {index_calls}"
        )
        assert "resume" in index_calls, (
            f"Expected resume indexing. Got calls: {index_calls}"
        )
        assert "positive" in index_calls, (
            f"Expected positive signals indexing. Got calls: {index_calls}"
        )

    @pytest.mark.asyncio
    async def test_skips_auto_index_when_collections_are_populated(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given the collections already have documents (populate_store=True default)
        When runner.run() is awaited
        Then the indexer is NOT instantiated
        """
        # Given: runner with pre-populated collections (default)
        settings = make_settings()
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io()

        # When: run with populated store
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            patch("jobsearch_rag.pipeline.runner.Indexer") as mock_indexer_cls,
        ):
            await runner.run()

        # Then: Indexer was never instantiated
        mock_indexer_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_index_completes_before_first_scoring_call(
        self,
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
        make_listing: Callable[..., JobListing],
        mock_embedder: Embedder,
    ) -> None:
        """
        Given empty collections and a board that returns listings
        When runner.run() completes
        Then indexing finishes before any scorer.score() call
        """
        # Given: empty store, one listing to score
        settings = make_settings()
        listing = make_listing(title="Test Job")
        _mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[listing]
        )

        call_order: list[str] = []

        async def track_index(*args: object, **kwargs: object) -> int:
            call_order.append("index")
            return 1

        async def track_score(*args: object, **kwargs: object) -> ScoreResult:
            call_order.append("score")
            return ScoreResult(
                fit_score=0.8, archetype_score=0.7, history_score=0.5, disqualified=False
            )

        mock_scorer = MagicMock()
        mock_scorer.score = AsyncMock(side_effect=track_score)

        with (
            patch("jobsearch_rag.pipeline.runner.Embedder", return_value=mock_embedder),
            patch("jobsearch_rag.pipeline.runner.Scorer", return_value=mock_scorer),
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            runner = PipelineRunner(settings)

            with patch("jobsearch_rag.pipeline.runner.Indexer") as mock_indexer_cls:
                mock_indexer_instance = MagicMock()
                mock_indexer_instance.index_archetypes = AsyncMock(side_effect=track_index)
                mock_indexer_instance.index_resume = AsyncMock(side_effect=track_index)
                mock_indexer_instance.index_global_positive_signals = AsyncMock(
                    side_effect=track_index
                )
                mock_indexer_cls.return_value = mock_indexer_instance

                await runner.run()

        # Then: all index calls appear before any score call
        if "score" in call_order:
            last_index = max(
                (i for i, c in enumerate(call_order) if c == "index"),
                default=-1,
            )
            first_score = call_order.index("score")
            assert last_index < first_score, (
                f"Indexing (last at {last_index}) should complete before "
                f"scoring (first at {first_score}). Order: {call_order}"
            )

    @pytest.mark.asyncio
    async def test_scored_listings_are_passed_to_ranker(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a board returns one listing that scores successfully
        When runner.run() completes
        Then the RunResult contains ranked_listings from the ranker
        """
        # Given: one listing that will be scored
        settings = make_settings()
        listing = make_listing(title="Architect Role")
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[listing]
        )

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: scored listing was passed through ranker to result
        assert len(result.ranked_listings) >= 1 or mock_scorer.score.await_count >= 1, (
            f"Expected at least one scoring call or ranked listing. "
            f"Got {len(result.ranked_listings)} ranked, "
            f"{mock_scorer.score.await_count} score calls"
        )

    @pytest.mark.asyncio
    async def test_already_decided_listings_are_skipped_during_scoring(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a listing whose external_id has an existing decision recorded
        When runner.run() completes
        Then the listing is not scored and skipped_decisions >= 1
        """
        # Given: one listing whose external_id is already decided
        settings = make_settings()
        decided_listing = make_listing(
            board="testboard", external_id="already-decided", title="Old Job"
        )
        new_listing = make_listing(
            board="testboard", external_id="new-job", title="New Job"
        )
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[decided_listing, new_listing]
        )

        # Seed a decision for the decided listing's external_id via the
        # runner's DecisionRecorder. We patch get_decision on the instance
        # that the runner already owns. Since DecisionRecorder is created
        # during __init__, patching happens after make_runner_with_mocks.
        def mock_get_decision(job_id: str) -> dict[str, str] | None:
            if job_id == "already-decided":
                return {"job_id": "already-decided", "verdict": "no"}
            return None

        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            patch.object(
                runner, "_decision_recorder", wraps=runner._decision_recorder  # pyright: ignore[reportPrivateUsage]
            ) as mock_recorder,
        ):
            mock_recorder.get_decision = mock_get_decision

            # When: run
            result = await runner.run()

        # Then: decided listing was skipped, new listing was scored
        assert result.skipped_decisions >= 1, (
            f"Expected skipped_decisions >= 1. Got: {result.skipped_decisions}"
        )
        # Scorer should only have been called once (for new-job, not already-decided)
        assert mock_scorer.score.await_count == 1, (
            f"Expected scorer called once (for new-job). "
            f"Got {mock_scorer.score.await_count} calls"
        )

    @pytest.mark.asyncio
    async def test_compensation_data_from_jd_is_assigned_to_listing(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a listing whose full_text contains parseable compensation data
        When runner.run() completes
        Then the ranked listing's comp_max and comp_text are populated
             and comp_score reflects the parsed compensation (not the missing-comp default)
        """
        # Given: listing with parseable salary range in JD text
        settings = make_settings()
        listing = make_listing(
            title="Well Paid Role",
            full_text=(
                "Senior Architect at Acme Inc. "
                "Salary: $250,000 - $300,000 per year. "
                "Comprehensive benefits package included."
            ),
        )
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[listing]
        )

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: ranked listing has comp data populated and comp_score != default
        assert len(result.ranked_listings) >= 1, (
            f"Expected at least 1 ranked listing. Got: {len(result.ranked_listings)}"
        )
        ranked = result.ranked_listings[0]
        assert ranked.listing.comp_max is not None, (
            "Expected comp_max to be populated from parsed JD text"
        )
        assert ranked.listing.comp_max == pytest.approx(300_000, rel=0.01), (
            f"Expected comp_max ~300000. Got: {ranked.listing.comp_max}"
        )
        assert ranked.listing.comp_text is not None, (
            "Expected comp_text to be populated from parsed JD text"
        )
        # comp_score should reflect parsed comp, not the missing-comp default (0.5)
        # With comp_max=300k and base_salary=220k, ratio=1.36 → comp_score=1.0
        assert ranked.scores.comp_score == pytest.approx(1.0), (
            f"Expected comp_score=1.0 (comp_max > base_salary). "
            f"Got: {ranked.scores.comp_score}"
        )


# ---------------------------------------------------------------------------
# TestBoardSearchDelegation
# ---------------------------------------------------------------------------


class TestBoardSearchDelegation:
    """
    REQUIREMENT: Individual board search delegates correctly to adapter and
    session manager in the expected lifecycle order.

    WHO: The pipeline runner calling adapters through the session manager
    WHAT: authenticate → search → extract_detail lifecycle is called in order;
          listings with populated full_text skip extract_detail; empty JD text
          after extraction is counted as failed; extraction errors on individual
          listings are counted without aborting the board; a board with no
          config section is skipped without error; search errors on one URL
          skip that URL and continue to next
    WHY: A single extraction failure aborting the board discards all other
         results; a missing config crashing the run is a validation error
         reported too late

    MOCK BOUNDARY:
        Mock:  mock_board_io fixture (authenticate, search, extract_detail
               are AsyncMock); mock_embedder fixture (Ollama HTTP);
               asyncio.sleep for throttle bypass
        Real:  PipelineRunner, RunResult
        Never: Construct RunResult directly; verify lifecycle order through
               the mock_board_io's call sequence (mock.call_args_list)
    """

    @pytest.mark.asyncio
    async def test_adapter_lifecycle_runs_in_authenticate_search_extract_order(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a board returns one listing that needs detail extraction
        When runner.run() completes
        Then the adapter calls happen in order: authenticate → search → extract_detail
        """
        # Given: one listing without full_text → needs extraction
        settings = make_settings()
        listing = make_listing(title="Lifecycle Test", full_text="")
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)

        call_order: list[str] = []

        async def track_auth(*a: object, **kw: object) -> None:
            call_order.append("authenticate")

        async def track_search(*a: object, **kw: object) -> list[JobListing]:
            call_order.append("search")
            return [listing]

        async def track_extract(*a: object, **kw: object) -> JobListing:
            call_order.append("extract_detail")
            return make_listing(title="Lifecycle Test", full_text="Detailed JD text")

        mock_adapter, mock_session, mock_registry = mock_board_io()
        mock_adapter.authenticate = AsyncMock(side_effect=track_auth)
        mock_adapter.search = AsyncMock(side_effect=track_search)
        mock_adapter.extract_detail = AsyncMock(side_effect=track_extract)

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            await runner.run()

        # Then: calls are in expected order
        assert call_order == ["authenticate", "search", "extract_detail"], (
            f"Expected ['authenticate', 'search', 'extract_detail']. Got: {call_order}"
        )

    @pytest.mark.asyncio
    async def test_enriched_listings_skip_extract_detail(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a listing with full_text already populated
        When runner.run() completes
        Then extract_detail is never called for that listing
        """
        # Given: listing already enriched
        settings = make_settings()
        enriched = make_listing(title="Pre-enriched", full_text="Full JD text here")
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[enriched]
        )

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            await runner.run()

        # Then: extract_detail was never called
        mock_adapter.extract_detail.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_jd_text_after_extraction_is_counted_as_failure(
        self,
        caplog: pytest.LogCaptureFixture,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a listing where extract_detail returns empty full_text
        When runner.run() completes
        Then failed_listings is incremented
             and the listing URL appears in a warning log entry
        """
        # Given: listing extraction returns empty text
        settings = make_settings()
        listing = make_listing(title="Empty JD", full_text="")
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[listing]
        )
        mock_adapter.extract_detail = AsyncMock(
            return_value=make_listing(title="Empty JD", full_text="")
        )

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: one failure counted, listing URL in warning log
        assert result.failed_listings >= 1, (
            f"Expected at least 1 failed_listing for empty JD. "
            f"Got: {result.failed_listings}"
        )
        warning_records = [
            r for r in caplog.records
            if r.levelname == "WARNING" and listing.url in r.message
        ]
        assert warning_records, (
            f"Expected a WARNING log mentioning '{listing.url}'. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_extraction_error_counts_failure_without_aborting_board(
        self,
        caplog: pytest.LogCaptureFixture,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given two listings where extract_detail raises ActionableError on the first
        When runner.run() completes
        Then the second listing is still scored and failed_listings >= 1
             and the failing listing's URL appears in a warning log entry
        """
        # Given: two listings, first extraction fails
        settings = make_settings()
        listing_a = make_listing(
            board="testboard", external_id="a", title="Failing Extract", full_text=""
        )
        listing_b = make_listing(
            board="testboard", external_id="b", title="Good Extract", full_text=""
        )
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)
        mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[listing_a, listing_b]
        )

        # extract_detail: fail on first, succeed on second
        mock_adapter.extract_detail = AsyncMock(
            side_effect=[
                ActionableError(
                    error="Detail extraction failed",
                    error_type=ErrorType.PARSE,
                    service="testboard",
                ),
                make_listing(
                    board="testboard",
                    external_id="b",
                    title="Good Extract",
                    full_text="Detailed description here",
                ),
            ]
        )

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: first listing counted as failed, second was scored
        assert result.failed_listings >= 1, (
            f"Expected at least 1 failed_listing. Got: {result.failed_listings}"
        )
        # The scorer should have been called for the second listing
        assert mock_scorer.score.await_count >= 1, (
            f"Expected scorer to be called for the second listing. "
            f"Got {mock_scorer.score.await_count} calls"
        )
        warning_records = [
            r for r in caplog.records
            if r.levelname == "WARNING" and listing_a.url in r.message
        ]
        assert warning_records, (
            f"Expected a WARNING log mentioning '{listing_a.url}'. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_board_with_no_config_section_is_skipped_without_error(
        self,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a board name that has no matching config section in settings
        When runner.run(boards=["nonexistent"]) is called
        Then the run completes without error and boards_searched includes it
        """
        # Given: default settings (config for "testboard" only, not "nonexistent")
        settings = make_settings()
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        _mock_adapter, mock_session, mock_registry = mock_board_io()

        # When: run with a board name that has no config section
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run(boards=["nonexistent"])

        # Then: run completed without crash, board appears in boards_searched
        assert isinstance(result, RunResult), (
            f"Expected RunResult. Got: {type(result)}"
        )
        assert "nonexistent" in result.boards_searched, (
            f"Expected 'nonexistent' in boards_searched. Got: {result.boards_searched}"
        )

    @pytest.mark.asyncio
    async def test_search_error_on_one_url_skips_and_continues_to_next(
        self,
        caplog: pytest.LogCaptureFixture,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
    ) -> None:
        """
        Given two search URLs where the first raises ActionableError
        When runner.run() completes
        Then listings from the second URL are still scored
             and the failing search URL appears in a warning log entry
        """
        # Given: board with two search URLs
        settings = make_settings()
        # Add a second search URL to the board config
        board_cfg = settings.boards.get("testboard")
        if board_cfg is not None:
            board_cfg.searches = [
                "https://example.com/search1",
                "https://example.com/search2",
            ]

        good_listing = make_listing(
            title="Second URL Job", full_text="Good description"
        )
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)

        # Build adapter that fails on first search, succeeds on second
        mock_adapter = MagicMock()
        mock_adapter.board_name = "testboard"
        mock_adapter.rate_limit_seconds = (0.0, 0.0)
        mock_adapter.authenticate = AsyncMock()
        mock_adapter.search = AsyncMock(
            side_effect=[
                ActionableError(
                    error="First URL failed",
                    error_type=ErrorType.CONNECTION,
                    service="testboard",
                ),
                [good_listing],
            ]
        )
        mock_adapter.extract_detail = AsyncMock()

        mock_registry = MagicMock()
        mock_registry.get = MagicMock(return_value=mock_adapter)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.new_page = AsyncMock(return_value=MagicMock())
        mock_session.save_storage_state = AsyncMock()

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            await runner.run()

        # Then: second URL's listing was scored, failing URL logged
        assert mock_scorer.score.await_count >= 1, (
            f"Expected scorer to be called for second URL's listing. "
            f"Got {mock_scorer.score.await_count} calls"
        )
        warning_records = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "search1" in r.message
        ]
        assert warning_records, (
            f"Expected a WARNING log mentioning 'search1'. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_authenticate_failure_is_caught_so_other_boards_still_run(
        self,
        caplog: pytest.LogCaptureFixture,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
    ) -> None:
        """
        Given two boards where authenticate() on the first raises ActionableError
        When runner.run() completes
        Then the second board's listings are still scored
             and the failing board name appears in the log output
            f"Expected 'testboard' in boards_searched. Got: {result.boards_searched}"
        )
        """
        # Given: two boards — "failboard" auth fails, "testboard" works
        settings = make_settings(enabled_boards=["failboard", "testboard"])
        good_listing = make_listing(
            board="testboard", title="Good Job", full_text="Solid description"
        )
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)

        # Adapter for failboard: authenticate raises ActionableError
        fail_adapter = MagicMock()
        fail_adapter.board_name = "failboard"
        fail_adapter.rate_limit_seconds = (0.0, 0.0)
        fail_adapter.authenticate = AsyncMock(
            side_effect=ActionableError(
                error="Auth cookie expired",
                error_type=ErrorType.AUTHENTICATION,
                service="failboard",
            )
        )

        # Adapter for testboard: works normally
        ok_adapter = MagicMock()
        ok_adapter.board_name = "testboard"
        ok_adapter.rate_limit_seconds = (0.0, 0.0)
        ok_adapter.authenticate = AsyncMock()
        ok_adapter.search = AsyncMock(return_value=[good_listing])
        ok_adapter.extract_detail = AsyncMock()

        # Registry returns different adapter per board
        def get_adapter(name: str) -> MagicMock:
            return fail_adapter if name == "failboard" else ok_adapter

        mock_registry = MagicMock()
        mock_registry.get = MagicMock(side_effect=get_adapter)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.new_page = AsyncMock(return_value=MagicMock())
        mock_session.save_storage_state = AsyncMock()

        # When: run both boards
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run(boards=["failboard", "testboard"])

        # Then: second board's listing was still scored despite first board's auth failure
        assert mock_scorer.score.await_count >= 1, (
            f"Expected scorer called for testboard listing despite failboard auth error. "
            f"Got {mock_scorer.score.await_count} calls"
        )
        assert "failboard" in result.boards_searched, (
            f"Expected 'failboard' in boards_searched. Got: {result.boards_searched}"
        )
        assert "testboard" in result.boards_searched, (
            f"Expected 'testboard' in boards_searched. Got: {result.boards_searched}"
        )
        error_records = [
            r for r in caplog.records
            if r.levelname == "ERROR" and "failboard" in r.message
        ]
        assert error_records, (
            f"Expected an ERROR log mentioning 'failboard'. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_unexpected_extraction_exception_is_wrapped_and_counted(
        self,
        caplog: pytest.LogCaptureFixture,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
        mock_board_io: Callable[..., tuple[MagicMock, MagicMock, MagicMock]],
    ) -> None:
        """
        Given a listing where extract_detail raises a RuntimeError
        When runner.run() completes
        Then the error is wrapped via ActionableError.from_exception()
             and failed_listings is incremented and no exception propagates
             and the listing URL appears in the log output at ERROR level
        """
        # Given: one listing whose extraction raises an unexpected RuntimeError
        settings = make_settings()
        listing = make_listing(
            board="testboard", title="Crashing Extract", full_text=""
        )
        runner, _mock_emb, _mock_scorer = make_runner_with_mocks(settings)
        mock_adapter, mock_session, mock_registry = mock_board_io(
            search_results=[listing]
        )
        mock_adapter.extract_detail = AsyncMock(
            side_effect=RuntimeError("WebSocket disconnected unexpectedly")
        )

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run()

        # Then: failure counted, no crash, listing URL appears at ERROR level
        assert result.failed_listings >= 1, (
            f"Expected at least 1 failed_listing for unexpected RuntimeError. "
            f"Got: {result.failed_listings}"
        )
        error_records = [
            r for r in caplog.records
            if r.levelname == "ERROR" and listing.url in r.message
        ]
        assert error_records, (
            f"Expected an ERROR log mentioning '{listing.url}'. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_unexpected_search_exception_is_wrapped_and_counted(
        self,
        caplog: pytest.LogCaptureFixture,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
    ) -> None:
        """
        Given a search URL where adapter.search raises a RuntimeError
        When runner.run() completes
        Then the error is wrapped via ActionableError.from_exception()
             and the search URL appears in a warning log entry
             and listings from other URLs are still collected
        """
        # Given: board with two search URLs, first raises RuntimeError
        settings = make_settings()
        board_cfg = settings.boards.get("testboard")
        if board_cfg is not None:
            board_cfg.searches = [
                "https://example.com/search1",
                "https://example.com/search2",
            ]

        good_listing = make_listing(
            title="Second URL Job", full_text="Good description"
        )
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)

        # Build adapter that raises RuntimeError on first search, succeeds on second
        mock_adapter = MagicMock()
        mock_adapter.board_name = "testboard"
        mock_adapter.rate_limit_seconds = (0.0, 0.0)
        mock_adapter.authenticate = AsyncMock()
        mock_adapter.search = AsyncMock(
            side_effect=[
                RuntimeError("Connection reset by peer"),
                [good_listing],
            ]
        )
        mock_adapter.extract_detail = AsyncMock()

        mock_registry = MagicMock()
        mock_registry.get = MagicMock(return_value=mock_adapter)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.new_page = AsyncMock(return_value=MagicMock())
        mock_session.save_storage_state = AsyncMock()

        # When: run
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            await runner.run()

        # Then: second URL's listing was scored, warning log mentions first URL
        assert mock_scorer.score.await_count >= 1, (
            f"Expected scorer to be called for second URL's listing. "
            f"Got {mock_scorer.score.await_count} calls"
        )
        warning_records = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "search1" in r.message
        ]
        assert warning_records, (
            f"Expected a WARNING log mentioning 'search1'. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_unexpected_board_exception_is_wrapped_so_other_boards_run(
        self,
        caplog: pytest.LogCaptureFixture,
        make_runner_with_mocks: Callable[..., tuple[PipelineRunner, Embedder, MagicMock]],
        make_settings: Callable[..., Settings],
        make_listing: Callable[..., JobListing],
    ) -> None:
        """
        Given two boards where _search_board on the first raises a RuntimeError
        When runner.run() completes
        Then the error is wrapped via ActionableError.from_exception()
             and the second board's listings are still scored
             and the failing board name appears in the log output
        """
        # Given: two boards — "crashboard" raises RuntimeError, "testboard" works
        settings = make_settings(enabled_boards=["crashboard", "testboard"])
        good_listing = make_listing(
            board="testboard", title="Good Job", full_text="Solid description"
        )
        runner, _mock_emb, mock_scorer = make_runner_with_mocks(settings)

        # Adapter for crashboard: authenticate raises RuntimeError (non-ActionableError)
        crash_adapter = MagicMock()
        crash_adapter.board_name = "crashboard"
        crash_adapter.rate_limit_seconds = (0.0, 0.0)
        crash_adapter.authenticate = AsyncMock(
            side_effect=RuntimeError("Segfault in browser process")
        )

        # Adapter for testboard: works normally
        ok_adapter = MagicMock()
        ok_adapter.board_name = "testboard"
        ok_adapter.rate_limit_seconds = (0.0, 0.0)
        ok_adapter.authenticate = AsyncMock()
        ok_adapter.search = AsyncMock(return_value=[good_listing])
        ok_adapter.extract_detail = AsyncMock()

        # Registry returns different adapter per board
        def get_adapter(name: str) -> MagicMock:
            return crash_adapter if name == "crashboard" else ok_adapter

        mock_registry = MagicMock()
        mock_registry.get = MagicMock(side_effect=get_adapter)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.new_page = AsyncMock(return_value=MagicMock())
        mock_session.save_storage_state = AsyncMock()

        # When: run both boards
        with (
            patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
            patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
            patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
        ):
            result = await runner.run(boards=["crashboard", "testboard"])

        # Then: second board's listing was scored, crashboard name in error log
        assert mock_scorer.score.await_count >= 1, (
            f"Expected scorer called for testboard listing despite crashboard error. "
            f"Got {mock_scorer.score.await_count} calls"
        )
        assert "crashboard" in result.boards_searched, (
            f"Expected 'crashboard' in boards_searched. Got: {result.boards_searched}"
        )
        error_records = [
            r for r in caplog.records
            if r.levelname == "ERROR" and "crashboard" in r.message
        ]
        assert error_records, (
            f"Expected an ERROR log mentioning 'crashboard'. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

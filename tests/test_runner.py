"""Pipeline runner tests — orchestration, error handling, board delegation.

Maps to BDD specs: TestPipelineOrchestration, TestBoardSearchDelegation
"""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.config import (
    BoardConfig,
    ChromaConfig,
    OllamaConfig,
    OutputConfig,
    ScoringConfig,
    Settings,
)
from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.pipeline.runner import PipelineRunner, RunResult
from jobsearch_rag.rag.scorer import ScoreResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_FAKE = [0.1, 0.2, 0.3, 0.4, 0.5]


def _make_settings(
    tmpdir: str,
    enabled_boards: list[str] | None = None,
    overnight_boards: list[str] | None = None,
) -> Settings:
    """Create a Settings with temp ChromaDB dir and configurable boards."""
    boards = enabled_boards or ["testboard"]
    board_configs = {}
    for name in boards:
        board_configs[name] = BoardConfig(
            name=name,
            searches=[f"https://{name}.com/search"],
            max_pages=1,
            headless=True,
        )
    # Also add overnight board configs
    for name in (overnight_boards or []):
        if name not in board_configs:
            board_configs[name] = BoardConfig(
                name=name,
                searches=[f"https://{name}.com/search"],
                max_pages=1,
                headless=False,
            )
    return Settings(
        enabled_boards=boards,
        overnight_boards=overnight_boards or [],
        boards=board_configs,
        scoring=ScoringConfig(),
        ollama=OllamaConfig(),
        output=OutputConfig(),
        chroma=ChromaConfig(persist_dir=tmpdir),
    )


def _make_listing(
    board: str = "testboard",
    external_id: str = "1",
    title: str = "Staff Architect",
) -> JobListing:
    return JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company="Acme Corp",
        location="Remote",
        url=f"https://{board}.com/{external_id}",
        full_text="A detailed job description for a staff architect role.",
    )


def _make_runner_with_mocks(
    settings: Settings,
) -> tuple[PipelineRunner, MagicMock, MagicMock, MagicMock]:
    """Create a PipelineRunner with mocked Embedder, Scorer, and Store.

    Returns (runner, mock_embedder, mock_scorer, mock_store).
    """
    mock_embedder = MagicMock()
    mock_embedder.health_check = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=EMBED_FAKE)

    mock_store = MagicMock()

    mock_scorer = MagicMock()
    mock_scorer.score = AsyncMock(
        return_value=ScoreResult(
            fit_score=0.8,
            archetype_score=0.7,
            history_score=0.5,
            disqualified=False,
        )
    )

    with (
        patch("jobsearch_rag.pipeline.runner.Embedder", return_value=mock_embedder),
        patch("jobsearch_rag.pipeline.runner.VectorStore", return_value=mock_store),
        patch("jobsearch_rag.pipeline.runner.Scorer", return_value=mock_scorer),
    ):
        runner = PipelineRunner(settings)

    # Replace internal references (they were set in __init__)
    runner._embedder = mock_embedder
    runner._scorer = mock_scorer
    runner._store = mock_store

    return runner, mock_embedder, mock_scorer, mock_store


def _mock_board_io(
    *,
    search_results: list[JobListing] | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create mocked I/O boundaries for board search (adapter, session, registry).

    Returns ``(mock_adapter, mock_session, mock_registry)`` for use
    with ``patch()``.  The adapter returns *search_results* (default: empty).
    """
    mock_adapter = MagicMock()
    mock_adapter.board_name = "testboard"
    mock_adapter.rate_limit_seconds = (0.0, 0.0)
    mock_adapter.authenticate = AsyncMock()
    mock_adapter.search = AsyncMock(
        return_value=search_results if search_results is not None else []
    )
    mock_adapter.extract_detail = AsyncMock()

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.new_page = AsyncMock(return_value=MagicMock())
    mock_session.save_storage_state = AsyncMock()

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_adapter

    return mock_adapter, mock_session, mock_registry


# ---------------------------------------------------------------------------
# TestPipelineOrchestration
# ---------------------------------------------------------------------------


class TestPipelineOrchestration:
    """REQUIREMENT: The pipeline runner executes steps in correct order with proper error handling.

    WHO: The operator running a search; downstream consumers of RunResult
    WHAT: Ollama health check runs before any browser work; enabled boards
          are searched when no explicit board list is given; overnight mode
          includes overnight-only boards; a board that fails entirely does
          not abort other boards; scoring failures increment the failed count
          without aborting; empty results return a valid RunResult with zero
          ranked listings
    WHY: A health check after browser work wastes time; a single board
         failure aborting the entire run discards valid results from other
         boards; an invalid RunResult crashes exporters downstream
    """

    async def test_health_check_runs_before_board_search(self) -> None:
        """Ollama health check is the first async step — before any browser work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, mock_embedder, _, _ = _make_runner_with_mocks(settings)

            call_order: list[str] = []
            original_health_check = mock_embedder.health_check

            async def _tracked_health_check() -> None:
                call_order.append("health_check")
                return await original_health_check()

            mock_embedder.health_check = _tracked_health_check

            mock_adapter, mock_session, mock_registry = _mock_board_io()
            original_auth = mock_adapter.authenticate

            async def _tracked_auth(page: object) -> None:
                call_order.append("board_io")
                return await original_auth(page)

            mock_adapter.authenticate = _tracked_auth

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                await runner.run()

            assert call_order[0] == "health_check"

    async def test_defaults_to_enabled_boards_when_none_specified(self) -> None:
        """When boards=None, all enabled_boards from settings are searched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir, enabled_boards=["board_a", "board_b"])
            runner, _, _, _ = _make_runner_with_mocks(settings)

            _, mock_session, mock_registry = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                result = await runner.run(boards=None)

            searched = {call.args[0] for call in mock_registry.get.call_args_list}
            assert searched == {"board_a", "board_b"}
            assert set(result.boards_searched) == {"board_a", "board_b"}

    async def test_explicit_boards_override_enabled_boards(self) -> None:
        """An explicit boards list overrides the enabled_boards in settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir, enabled_boards=["board_a", "board_b"])
            runner, _, _, _ = _make_runner_with_mocks(settings)

            _, mock_session, mock_registry = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                await runner.run(boards=["board_a"])

            searched = [call.args[0] for call in mock_registry.get.call_args_list]
            assert searched == ["board_a"]

    async def test_overnight_mode_includes_overnight_boards(self) -> None:
        """Overnight mode adds overnight-only boards alongside the enabled boards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(
                tmpdir,
                enabled_boards=["board_a"],
                overnight_boards=["linkedin"],
            )
            runner, _, _, _ = _make_runner_with_mocks(settings)

            _, mock_session, mock_registry = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                await runner.run(overnight=True)

            searched = {call.args[0] for call in mock_registry.get.call_args_list}
            assert "board_a" in searched
            assert "linkedin" in searched

    async def test_board_failure_does_not_abort_other_boards(self) -> None:
        """An ActionableError from one board does not prevent other boards from running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir, enabled_boards=["failing_board", "good_board"])
            runner, _, _, _ = _make_runner_with_mocks(settings)

            # Failing board: authenticate raises ActionableError
            failing_adapter = MagicMock()
            failing_adapter.board_name = "failing_board"
            failing_adapter.rate_limit_seconds = (0.0, 0.0)
            failing_adapter.authenticate = AsyncMock(
                side_effect=ActionableError(
                    error="Board failed",
                    error_type=ErrorType.CONNECTION,
                    service="failing_board",
                )
            )

            # Good board: returns a listing with populated full_text
            good_adapter = MagicMock()
            good_adapter.board_name = "good_board"
            good_adapter.rate_limit_seconds = (0.0, 0.0)
            good_adapter.authenticate = AsyncMock()
            good_adapter.search = AsyncMock(return_value=[_make_listing(board="good_board")])
            good_adapter.extract_detail = AsyncMock()

            mock_registry = MagicMock()
            mock_registry.get.side_effect = (
                lambda name: failing_adapter if name == "failing_board" else good_adapter
            )

            _, mock_session, _ = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                result = await runner.run()

            assert "good_board" in result.boards_searched
            assert "failing_board" in result.boards_searched
            # good_board's listing should have been scored
            assert result.summary.total_found >= 1

    async def test_scoring_failure_increments_failed_count(self) -> None:
        """A scoring failure on an individual listing increments failed_listings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, mock_scorer, _ = _make_runner_with_mocks(settings)

            listing = _make_listing()
            _, mock_session, mock_registry = _mock_board_io(search_results=[listing])

            # Scorer throws on every call
            mock_scorer.score = AsyncMock(
                side_effect=ActionableError(
                    error="Scoring failed",
                    error_type=ErrorType.EMBEDDING,
                    service="scorer",
                )
            )

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                result = await runner.run()

            assert result.failed_listings >= 1

    async def test_empty_results_return_valid_run_result(self) -> None:
        """When no listings are collected, a valid RunResult is returned with zero ranked listings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            _, mock_session, mock_registry = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                result = await runner.run()

            assert isinstance(result, RunResult)
            assert result.ranked_listings == []
            assert result.boards_searched == ["testboard"]

    async def test_scored_listings_are_passed_to_ranker(self) -> None:
        """Successfully scored listings are passed through the ranker for fusion and dedup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            listing = _make_listing()
            _, mock_session, mock_registry = _mock_board_io(search_results=[listing])

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                result = await runner.run()

            # The listing should have been scored and ranked
            assert result.summary.total_found == 1
            assert result.summary.total_scored == 1


# ---------------------------------------------------------------------------
# TestBoardSearchDelegation
# ---------------------------------------------------------------------------


class TestBoardSearchDelegation:
    """REQUIREMENT: Individual board search delegates correctly to adapter and session manager.

    WHO: The pipeline runner calling adapters through the session manager
    WHAT: A board with no config section is skipped without error;
          the adapter's authenticate → search → extract_detail lifecycle
          is called in order; empty JD text from extraction is counted
          as a failure; extraction errors on individual listings are counted
          without aborting the board
    WHY: A missing config crashing the run is a config error reported too late;
         a single extraction failure aborting the board discards all other
         results from that board's search
    """

    async def test_board_with_no_config_section_is_skipped(self) -> None:
        """A board name not in settings.boards returns empty results without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            # AdapterRegistry.get succeeds, but board has no config section
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.new_page = AsyncMock(return_value=MagicMock())
            mock_session.save_storage_state = AsyncMock()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry") as mock_registry,
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                mock_registry.get.return_value = MagicMock()
                result = await runner.run(boards=["nonexistent_board"])

            assert result.ranked_listings == []
            assert result.failed_listings == 0

    async def test_adapter_lifecycle_runs_in_order(self) -> None:
        """authenticate → search → extract_detail is called in correct order.

        When search() returns listings with empty full_text, the runner
        calls extract_detail.  When full_text is already populated (e.g.
        via click-through), extract_detail is skipped.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            call_order: list[str] = []
            # Listing with EMPTY full_text — runner should call extract_detail
            listing = JobListing(
                board="testboard",
                external_id="1",
                title="Staff Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/1",
                full_text="",
            )

            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)

            async def _auth(page: object) -> None:
                call_order.append("authenticate")

            async def _search(page: object, url: str, max_pages: int = 1) -> list[JobListing]:
                call_order.append("search")
                return [listing]

            async def _extract(page: object, lst: JobListing) -> JobListing:
                call_order.append("extract_detail")
                lst.full_text = "Extracted detail text."
                return lst

            mock_adapter.authenticate = _auth
            mock_adapter.search = _search
            mock_adapter.extract_detail = _extract

            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.new_page = AsyncMock(return_value=MagicMock())
            mock_session.save_storage_state = AsyncMock()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry") as mock_registry,
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                mock_registry.get.return_value = mock_adapter
                result = await runner.run(boards=["testboard"])

            assert call_order == ["authenticate", "search", "extract_detail"]
            assert result.summary.total_found == 1

    async def test_enriched_listings_skip_extract_detail(self) -> None:
        """Listings with full_text already populated skip extract_detail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            call_order: list[str] = []
            # Listing with POPULATED full_text (e.g. from click-through)
            listing = _make_listing()

            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)

            async def _auth(page: object) -> None:
                call_order.append("authenticate")

            async def _search(page: object, url: str, max_pages: int = 1) -> list[JobListing]:
                call_order.append("search")
                return [listing]

            async def _extract(page: object, lst: JobListing) -> JobListing:
                call_order.append("extract_detail")
                return lst

            mock_adapter.authenticate = _auth
            mock_adapter.search = _search
            mock_adapter.extract_detail = _extract

            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.new_page = AsyncMock(return_value=MagicMock())
            mock_session.save_storage_state = AsyncMock()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry") as mock_registry,
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                mock_registry.get.return_value = mock_adapter
                result = await runner.run(boards=["testboard"])

            # extract_detail should NOT have been called
            assert call_order == ["authenticate", "search"]
            assert result.summary.total_found == 1

    async def test_empty_jd_text_is_counted_as_failure(self) -> None:
        """A listing with empty full_text after extraction is excluded and counted as failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            # Search returns listing with empty full_text (needs extract_detail)
            search_listing = JobListing(
                board="testboard",
                external_id="1",
                title="Staff Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/1",
                full_text="",
            )
            # extract_detail returns whitespace-only (should be counted as failure)
            empty_listing = JobListing(
                board="testboard",
                external_id="1",
                title="Empty Role",
                company="Co",
                location="Remote",
                url="https://example.org/1",
                full_text="   ",  # whitespace-only
            )

            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)
            mock_adapter.authenticate = AsyncMock()
            mock_adapter.search = AsyncMock(return_value=[search_listing])
            mock_adapter.extract_detail = AsyncMock(return_value=empty_listing)

            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.new_page = AsyncMock(return_value=MagicMock())
            mock_session.save_storage_state = AsyncMock()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry") as mock_registry,
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                mock_registry.get.return_value = mock_adapter
                result = await runner.run(boards=["testboard"])

            assert result.ranked_listings == []
            assert result.failed_listings == 1

    async def test_extraction_error_counts_failure_without_aborting(self) -> None:
        """An extraction error on one listing doesn't prevent processing the next."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            # Both listings need extract_detail (empty full_text)
            listing_good = JobListing(
                board="testboard",
                external_id="good",
                title="Staff Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/good",
                full_text="",
            )
            listing_bad = JobListing(
                board="testboard",
                external_id="bad",
                title="Staff Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/bad",
                full_text="",
            )

            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)
            mock_adapter.authenticate = AsyncMock()
            mock_adapter.search = AsyncMock(return_value=[listing_bad, listing_good])

            async def _extract(page: object, lst: JobListing) -> JobListing:
                if lst.external_id == "bad":
                    raise ActionableError(
                        error="Parse failed",
                        error_type=ErrorType.PARSE,
                        service="testboard",
                    )
                lst.full_text = "Extracted detail for good listing."
                return lst

            mock_adapter.extract_detail = _extract

            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.new_page = AsyncMock(return_value=MagicMock())
            mock_session.save_storage_state = AsyncMock()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry") as mock_registry,
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                mock_registry.get.return_value = mock_adapter
                result = await runner.run(boards=["testboard"])

            assert result.summary.total_found == 1  # good listing survived
            assert result.ranked_listings[0].listing.external_id == "good"
            assert result.failed_listings == 1  # bad listing counted

    async def test_unexpected_exception_during_extraction_is_counted(self) -> None:
        """An unexpected (non-ActionableError) exception during extraction is caught and counted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            # Listing with empty full_text to trigger extract_detail
            listing = JobListing(
                board="testboard",
                external_id="1",
                title="Staff Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/1",
                full_text="",
            )

            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)
            mock_adapter.authenticate = AsyncMock()
            mock_adapter.search = AsyncMock(return_value=[listing])
            mock_adapter.extract_detail = AsyncMock(side_effect=RuntimeError("unexpected"))

            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.new_page = AsyncMock(return_value=MagicMock())
            mock_session.save_storage_state = AsyncMock()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry") as mock_registry,
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                mock_registry.get.return_value = mock_adapter
                result = await runner.run(boards=["testboard"])

            assert result.ranked_listings == []
            assert result.failed_listings == 1

    async def test_search_failure_skips_url_and_continues(self) -> None:
        """An ActionableError during search for one URL skips it and tries the next."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            # Add a second search URL to the board config
            settings.boards["testboard"].searches = [
                "https://testboard.com/search1",
                "https://testboard.com/search2",
            ]
            runner, _, _, _ = _make_runner_with_mocks(settings)

            listing = _make_listing()
            call_count = {"search": 0}

            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)
            mock_adapter.authenticate = AsyncMock()

            async def _search(page: object, url: str, max_pages: int = 1) -> list[JobListing]:
                call_count["search"] += 1
                if "search1" in url:
                    raise ActionableError(
                        error="Search failed", error_type=ErrorType.PARSE, service="testboard"
                    )
                return [listing]

            mock_adapter.search = _search
            mock_adapter.extract_detail = AsyncMock(return_value=listing)

            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.new_page = AsyncMock(return_value=MagicMock())
            mock_session.save_storage_state = AsyncMock()

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry") as mock_registry,
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                mock_registry.get.return_value = mock_adapter
                result = await runner.run(boards=["testboard"])

            assert call_count["search"] == 2  # both URLs attempted
            assert result.summary.total_found == 1  # second URL's listing survived


# ---------------------------------------------------------------------------
# TestAutoIndex (Phase 4a — auto-recovery)
# ---------------------------------------------------------------------------


class TestAutoIndex:
    """REQUIREMENT: Empty collections are auto-indexed before scoring begins.

    WHO: The operator running search after a reset or on first use
    WHAT: If the resume or role_archetypes collection is empty, the runner
          auto-indexes before scoring; if collections are already populated,
          no re-indexing occurs; auto-index uses the same embedder and store
          as the rest of the pipeline
    WHY: Failing all 180 listings because the operator forgot to run a
         separate index command is a predictable situation with an obvious
         recovery — the system should just do it
    """

    async def test_auto_indexes_when_resume_collection_is_empty(self) -> None:
        """An empty resume collection triggers auto-indexing before scoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, mock_store = _make_runner_with_mocks(settings)

            # Simulate empty collections
            mock_store.collection_count.return_value = 0

            mock_indexer = MagicMock()
            mock_indexer.index_resume = AsyncMock(return_value=5)
            mock_indexer.index_archetypes = AsyncMock(return_value=3)

            _, mock_session, mock_registry = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.Indexer", return_value=mock_indexer),
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                await runner.run()

            mock_indexer.index_resume.assert_awaited_once()
            mock_indexer.index_archetypes.assert_awaited_once()

    async def test_skips_auto_index_when_collections_are_populated(self) -> None:
        """Populated collections skip the auto-index step entirely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, mock_store = _make_runner_with_mocks(settings)

            # Simulate populated collections
            mock_store.collection_count.return_value = 10

            _, mock_session, mock_registry = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.Indexer") as mock_indexer_cls,
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                await runner.run()

            mock_indexer_cls.assert_not_called()

    async def test_auto_index_runs_before_scoring_begins(self) -> None:
        """Auto-indexing completes before the first scoring call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, mock_scorer, mock_store = _make_runner_with_mocks(settings)

            call_order: list[str] = []

            # First call: empty (triggers auto-index), subsequent calls: populated
            count_calls = 0

            def _mock_count(name: str) -> int:
                nonlocal count_calls
                count_calls += 1
                # First two calls (resume + archetypes check) return 0
                return 0 if count_calls <= 2 else 5

            mock_store.collection_count.side_effect = _mock_count

            mock_indexer = MagicMock()

            async def _mock_index_resume(path: str) -> int:
                call_order.append("index")
                return 5

            async def _mock_index_archetypes(path: str) -> int:
                return 3

            mock_indexer.index_resume = _mock_index_resume
            mock_indexer.index_archetypes = _mock_index_archetypes

            listing = _make_listing()

            _adapter, mock_session, mock_registry = _mock_board_io(
                search_results=[listing],
            )

            async def _mock_score(text: str) -> ScoreResult:
                call_order.append("score")
                return ScoreResult(
                    fit_score=0.8,
                    archetype_score=0.7,
                    history_score=0.5,
                    disqualified=False,
                )

            mock_scorer.score = _mock_score

            with (
                patch("jobsearch_rag.pipeline.runner.Indexer", return_value=mock_indexer),
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                await runner.run()

            assert call_order[0] == "index", "Auto-index must run before scoring"
            assert "score" in call_order, "Scoring should still run after auto-index"


    async def test_collection_empty_returns_true_when_store_raises(self) -> None:
        """GIVEN store.collection_count raises ActionableError
        WHEN _ensure_indexed checks collections
        THEN the collection is treated as empty and auto-indexing runs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, mock_store = _make_runner_with_mocks(settings)

            # collection_count raises ActionableError (not just returns 0)
            mock_store.collection_count.side_effect = ActionableError(
                error="Collection not found — table does not exist",
                error_type=ErrorType.INDEX,
                service="chromadb",
            )

            mock_indexer = MagicMock()
            mock_indexer.index_resume = AsyncMock(return_value=5)
            mock_indexer.index_archetypes = AsyncMock(return_value=3)

            _, mock_session, mock_registry = _mock_board_io()

            with (
                patch("jobsearch_rag.pipeline.runner.Indexer", return_value=mock_indexer),
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                await runner.run()

            # Both collections were treated as empty → auto-indexing ran
            mock_indexer.index_resume.assert_awaited_once()
            mock_indexer.index_archetypes.assert_awaited_once()


class TestCompEnrichment:
    """REQUIREMENT: Listings with salary text have comp fields populated after scoring.

    WHO: The scorer computing comp_score; the exporter showing salary data
    WHAT: When parse_compensation finds salary data in a listing's full_text,
          comp_min, comp_max, comp_source, and comp_text are set on the listing
    WHY: Missing comp enrichment would leave salary fields empty even when
         the JD contains salary data, making comp_score always neutral
    """

    async def test_listing_with_salary_text_gets_comp_fields_populated(self) -> None:
        """GIVEN a listing whose full_text contains a salary range
        WHEN the pipeline runner scores it
        THEN comp_min, comp_max, comp_source, and comp_text are set on the listing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, mock_store = _make_runner_with_mocks(settings)

            # Simulate populated collections to skip auto-index
            mock_store.collection_count.return_value = 10

            # Listing with salary info in full_text
            listing = JobListing(
                board="testboard",
                external_id="comp-1",
                title="Principal Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/comp-1",
                full_text="Principal Architect role. Salary: $180,000 - $220,000 per year.",
            )

            _mock_adapter, mock_session, mock_registry = _mock_board_io(
                search_results=[listing],
            )

            with (
                patch("jobsearch_rag.pipeline.runner.AdapterRegistry", mock_registry),
                patch("jobsearch_rag.pipeline.runner.SessionManager", return_value=mock_session),
                patch("jobsearch_rag.pipeline.runner.throttle", new_callable=AsyncMock),
            ):
                result = await runner.run()

            assert result.summary.total_found == 1
            assert len(result.ranked_listings) >= 0  # may be filtered by threshold
            # Verify comp enrichment happened on the listing object
            assert listing.comp_min == 180_000.0
            assert listing.comp_max == 220_000.0
            assert listing.comp_source == "employer"
            assert "$180,000" in listing.comp_text

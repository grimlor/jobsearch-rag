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

            # Mock _search_board to track call order
            call_order: list[str] = []
            original_health_check = mock_embedder.health_check

            async def _tracked_health_check() -> None:
                call_order.append("health_check")
                return await original_health_check()

            async def _tracked_search_board(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                call_order.append(f"search_{name}")
                return [], 0

            mock_embedder.health_check = _tracked_health_check
            runner._search_board = _tracked_search_board  # type: ignore[method-assign]

            await runner.run()

            assert call_order[0] == "health_check"

    async def test_defaults_to_enabled_boards_when_none_specified(self) -> None:
        """When boards=None, all enabled_boards from settings are searched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir, enabled_boards=["board_a", "board_b"])
            runner, _, _, _ = _make_runner_with_mocks(settings)

            searched: list[str] = []

            async def _mock_search(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                searched.append(name)
                return [], 0

            runner._search_board = _mock_search  # type: ignore[method-assign]

            result = await runner.run(boards=None)

            assert set(searched) == {"board_a", "board_b"}
            assert set(result.boards_searched) == {"board_a", "board_b"}

    async def test_explicit_boards_override_enabled_boards(self) -> None:
        """An explicit boards list overrides the enabled_boards in settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir, enabled_boards=["board_a", "board_b"])
            runner, _, _, _ = _make_runner_with_mocks(settings)

            searched: list[str] = []

            async def _mock_search(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                searched.append(name)
                return [], 0

            runner._search_board = _mock_search  # type: ignore[method-assign]

            await runner.run(boards=["board_a"])

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

            searched: list[str] = []

            async def _mock_search(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                searched.append(name)
                return [], 0

            runner._search_board = _mock_search  # type: ignore[method-assign]

            await runner.run(overnight=True)

            assert "board_a" in searched
            assert "linkedin" in searched

    async def test_board_failure_does_not_abort_other_boards(self) -> None:
        """An ActionableError from one board does not prevent other boards from running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir, enabled_boards=["failing_board", "good_board"])
            runner, _, _, _ = _make_runner_with_mocks(settings)

            async def _mock_search(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                if name == "failing_board":
                    raise ActionableError(
                        error="Board failed",
                        error_type=ErrorType.CONNECTION,
                        service=name,
                    )
                return [_make_listing(board=name)], 0

            runner._search_board = _mock_search  # type: ignore[method-assign]

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

            async def _mock_search(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                return [listing], 0

            runner._search_board = _mock_search  # type: ignore[method-assign]

            # Scorer throws on every call
            mock_scorer.score = AsyncMock(
                side_effect=ActionableError(
                    error="Scoring failed",
                    error_type=ErrorType.EMBEDDING,
                    service="scorer",
                )
            )

            result = await runner.run()

            assert result.failed_listings >= 1

    async def test_empty_results_return_valid_run_result(self) -> None:
        """When no listings are collected, a valid RunResult is returned with zero ranked listings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            async def _mock_search(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                return [], 0

            runner._search_board = _mock_search  # type: ignore[method-assign]

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

            async def _mock_search(name: str, *, overnight: bool = False) -> tuple[list[JobListing], int]:
                return [listing], 0

            runner._search_board = _mock_search  # type: ignore[method-assign]

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
            with patch(
                "jobsearch_rag.pipeline.runner.AdapterRegistry"
            ) as mock_registry:
                mock_registry.get.return_value = MagicMock()
                listings, failures = await runner._search_board("nonexistent_board")

            assert listings == []
            assert failures == 0

    async def test_adapter_lifecycle_runs_in_order(self) -> None:
        """authenticate → search → extract_detail is called in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            call_order: list[str] = []
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
                listings, _failures = await runner._search_board("testboard")

            assert call_order == ["authenticate", "search", "extract_detail"]
            assert len(listings) == 1

    async def test_empty_jd_text_is_counted_as_failure(self) -> None:
        """A listing with empty full_text after extraction is excluded and counted as failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            listing = _make_listing()
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
            mock_adapter.search = AsyncMock(return_value=[listing])
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
                listings, failures = await runner._search_board("testboard")

            assert len(listings) == 0
            assert failures == 1

    async def test_extraction_error_counts_failure_without_aborting(self) -> None:
        """An extraction error on one listing doesn't prevent processing the next."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            listing_good = _make_listing(external_id="good")
            listing_bad = _make_listing(external_id="bad")

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
                listings, failures = await runner._search_board("testboard")

            assert len(listings) == 1  # good listing survived
            assert listings[0].external_id == "good"
            assert failures == 1  # bad listing counted

    async def test_unexpected_exception_during_extraction_is_counted(self) -> None:
        """An unexpected (non-ActionableError) exception during extraction is caught and counted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner, _, _, _ = _make_runner_with_mocks(settings)

            listing = _make_listing()

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
                listings, failures = await runner._search_board("testboard")

            assert len(listings) == 0
            assert failures == 1

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
                listings, _failures = await runner._search_board("testboard")

            assert call_count["search"] == 2  # both URLs attempted
            assert len(listings) == 1  # second URL's listing survived

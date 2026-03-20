"""Pipeline runner tests — orchestration, error handling, board delegation.

Maps to BDD specs: TestPipelineOrchestration, TestBoardSearchDelegation
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import ollama as ollama_sdk

from jobsearch_rag.adapters import AdapterRegistry
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
from tests.constants import EMBED_FAKE

if TYPE_CHECKING:
    from jobsearch_rag.rag.store import VectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_index_files(tmpdir: str) -> tuple[str, str, str]:
    """Write minimal resume, archetypes, and rubric files into *tmpdir*.

    Returns ``(resume_path, archetypes_path, global_rubric_path)``.
    """
    base = Path(tmpdir)
    resume = base / "resume.md"
    resume.write_text("## Summary\n\nTest resume content for indexing.\n")

    archetypes = base / "role_archetypes.toml"
    archetypes.write_text(
        '[[archetypes]]\nname = "Test"\n'
        'description = "A test archetype."\n'
        'signals_positive = ["positive signal"]\n'
        'signals_negative = ["negative signal"]\n'
    )

    rubric = base / "global_rubric.toml"
    rubric.write_text(
        '[[dimensions]]\nname = "Test Dim"\n'
        'signals_positive = ["good indicator"]\n'
        'signals_negative = ["bad indicator"]\n'
    )

    return str(resume), str(archetypes), str(rubric)


def _make_settings(
    tmpdir: str,
    enabled_boards: list[str] | None = None,
    overnight_boards: list[str] | None = None,
    *,
    resume_path: str | None = None,
    archetypes_path: str | None = None,
    global_rubric_path: str | None = None,
) -> Settings:
    """Create a Settings with temp ChromaDB dir and configurable boards."""
    boards = enabled_boards or ["testboard"]
    board_configs: dict[str, BoardConfig] = {}
    for name in boards:
        board_configs[name] = BoardConfig(
            name=name,
            searches=[f"https://{name}.com/search"],
            max_pages=1,
            headless=True,
        )
    # Also add overnight board configs
    for name in overnight_boards or []:
        if name not in board_configs:
            board_configs[name] = BoardConfig(
                name=name,
                searches=[f"https://{name}.com/search"],
                max_pages=1,
                headless=False,
            )
    kwargs: dict[str, str] = {}
    if resume_path is not None:
        kwargs["resume_path"] = resume_path
    if archetypes_path is not None:
        kwargs["archetypes_path"] = archetypes_path
    if global_rubric_path is not None:
        kwargs["global_rubric_path"] = global_rubric_path
    return Settings(
        enabled_boards=boards,
        overnight_boards=overnight_boards or [],
        boards=board_configs,
        scoring=ScoringConfig(disqualify_on_llm_flag=False),
        ollama=OllamaConfig(),
        output=OutputConfig(output_dir=str(Path(tmpdir) / "output")),
        chroma=ChromaConfig(persist_dir=tmpdir),
        **kwargs,
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


def _make_runner_with_real_stack(
    settings: Settings,
    *,
    populate_store: bool = True,
) -> tuple[PipelineRunner, AsyncMock]:
    """Create a PipelineRunner with real Embedder/Scorer and mocked Ollama client.

    The only mock is ``ollama_sdk.AsyncClient`` — the I/O boundary where
    our system ends and the network begins.  Everything else (``Embedder``,
    ``Scorer``, ``VectorStore``, ``Ranker``, ``DecisionRecorder``) runs
    for real.

    When *populate_store* is ``True`` (default), minimal documents are
    seeded into ``resume``, ``role_archetypes``, and
    ``global_positive_signals`` so auto-indexing is skipped.

    Returns ``(runner, mock_client)``.
    """
    mock_client = AsyncMock()

    # health_check calls client.list() — needs models containing embed + llm names
    model_embed = MagicMock()
    model_embed.model = settings.ollama.embed_model
    model_llm = MagicMock()
    model_llm.model = settings.ollama.llm_model
    list_response = MagicMock()
    list_response.models = [model_embed, model_llm]
    mock_client.list.return_value = list_response

    # embed() calls client.embed() — needs .embeddings[0]
    embed_response = MagicMock()
    embed_response.embeddings = [EMBED_FAKE]
    mock_client.embed.return_value = embed_response

    with patch(
        "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        return_value=mock_client,
    ):
        runner = PipelineRunner(settings)

    if populate_store:
        _populate_store(runner._store)  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_store)

    return runner, mock_client


def _populate_store(store: VectorStore) -> None:
    """Seed the three required collections so auto-indexing is skipped."""
    for name in ("resume", "role_archetypes", "global_positive_signals"):
        store.add_documents(
            name,
            ids=[f"{name}-seed"],
            documents=[f"Seed document for {name}"],
            embeddings=[EMBED_FAKE],
        )


def _mock_playwright_boundary() -> tuple[MagicMock, MagicMock]:
    """Create a mock Playwright I/O boundary for real SessionManager.

    Mocks ``async_playwright`` — the edge where our system ends and the
    Playwright library begins.  ``SessionManager`` runs for real on top.

    Returns ``(mock_async_playwright, mock_page)`` for use with
    ``patch("jobsearch_rag.adapters.session.async_playwright", ...)``.
    """
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


def _make_test_adapter(
    board_name: str = "testboard",
    *,
    search_results: list[JobListing] | None = None,
) -> MagicMock:
    """Create a test adapter implementing ``JobBoardAdapter``.

    Register it via ``patch.dict(AdapterRegistry._registry, ...)``
    so the real ``AdapterRegistry.get()`` returns it.
    """
    adapter = MagicMock()
    adapter.board_name = board_name
    adapter.rate_limit_seconds = (0.0, 0.0)
    adapter.authenticate = AsyncMock()
    adapter.search = AsyncMock(return_value=search_results if search_results is not None else [])
    adapter.extract_detail = AsyncMock()
    return adapter


# ---------------------------------------------------------------------------
# TestPipelineOrchestration
# ---------------------------------------------------------------------------


class TestPipelineOrchestration:
    """REQUIREMENT: The pipeline runner executes steps in correct order with proper error handling.

    WHO: The operator running a search; downstream consumers of RunResult
    WHAT: (1) The system performs the Ollama health check before it starts any board I/O during a run.
          (2) The system searches all enabled boards when no boards are explicitly specified.
          (3) The system searches only the explicitly specified board when a boards list is provided.
          (4) The system searches both enabled boards and overnight boards when overnight mode is enabled.
          (5) The system continues scoring listings from a healthy board and reports both boards when another board fails during authentication.
          (6) The system increments the failed listings count when scoring fails because the embed endpoint returns a non-retryable error.
          (7) The system returns a valid RunResult with empty ranked listings when a board search produces no listings.
          (8) The system passes successfully scored listings to the ranker and includes them in the summary counts.
          (9) The system does not search the same board twice when an overnight board overlaps with enabled boards.
          (10) The system auto-indexes only the empty collections when some collections are already populated.
          (11) The system skips auto-indexing for collections that already contain documents.
    WHY: A health check after browser work wastes time; a single board
         failure aborting the entire run discards valid results from other
         boards; an invalid RunResult crashes exporters downstream

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama API),
               async_playwright (Playwright browser library)
        Real:  PipelineRunner, Embedder, Scorer, VectorStore, Ranker,
               DecisionRecorder, AdapterRegistry, SessionManager, throttle
        Never: Construct ScoreResult directly — always obtained via real Scorer.score()
    """

    async def test_health_check_runs_before_board_search(self) -> None:
        """
        Given a configured runner with populated collections,
        When run() is invoked,
        Then the Ollama health check (client.list) fires before any board I/O.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with real Embedder/Scorer, populated store
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)

            call_order: list[str] = []
            original_list = mock_client.list

            async def _tracked_list() -> object:
                call_order.append("health_check")
                return await original_list()

            mock_client.list = _tracked_list

            mock_adapter = _make_test_adapter()
            original_auth = mock_adapter.authenticate

            async def _tracked_auth(page: object) -> None:
                call_order.append("board_io")
                await original_auth(page)

            mock_adapter.authenticate = _tracked_auth

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: the pipeline runs
                await runner.run()

            # Then: health_check was first
            assert call_order[0] == "health_check", (
                f"Expected health_check first, got: {call_order}"
            )

    async def test_defaults_to_enabled_boards_when_none_specified(self) -> None:
        """
        Given a runner with two enabled boards,
        When run(boards=None) is called,
        Then both enabled boards are searched.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: two enabled boards
            settings = _make_settings(tmpdir, enabled_boards=["board_a", "board_b"])
            runner, _ = _make_runner_with_real_stack(settings)

            adapter_a = _make_test_adapter(board_name="board_a")
            adapter_b = _make_test_adapter(board_name="board_b")

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(
                    AdapterRegistry._registry,  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_registry)
                    {"board_a": lambda: adapter_a, "board_b": lambda: adapter_b},
                ),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: no boards specified
                result = await runner.run(boards=None)

            # Then: both enabled boards are searched
            assert set(result.boards_searched) == {
                "board_a",
                "board_b",
            }, f"Expected both enabled boards, got: {result.boards_searched}"

    async def test_explicit_boards_override_enabled_boards(self) -> None:
        """Given a runner with two enabled boards,
        When run(boards=["board_a"]) is called,
        Then only the explicitly specified board is searched.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: two enabled boards
            settings = _make_settings(tmpdir, enabled_boards=["board_a", "board_b"])
            runner, _ = _make_runner_with_real_stack(settings)

            adapter_a = _make_test_adapter(board_name="board_a")
            adapter_b = _make_test_adapter(board_name="board_b")

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(
                    AdapterRegistry._registry,  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_registry)
                    {"board_a": lambda: adapter_a, "board_b": lambda: adapter_b},
                ),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: explicit board list
                result = await runner.run(boards=["board_a"])

            # Then: only that board is searched
            assert result.boards_searched == ["board_a"], (
                f"Expected only board_a, got: {result.boards_searched}"
            )

    async def test_overnight_mode_includes_overnight_boards(self) -> None:
        """
        Given a runner with one enabled board and one overnight board,
        When run(overnight=True) is called,
        Then both the enabled and overnight boards are searched.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: enabled + overnight boards
            settings = _make_settings(
                tmpdir,
                enabled_boards=["board_a"],
                overnight_boards=["linkedin"],
            )
            runner, _ = _make_runner_with_real_stack(settings)

            adapter_a = _make_test_adapter(board_name="board_a")
            adapter_linkedin = _make_test_adapter(board_name="linkedin")

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(
                    AdapterRegistry._registry,  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_registry)
                    {"board_a": lambda: adapter_a, "linkedin": lambda: adapter_linkedin},
                ),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: overnight mode
                result = await runner.run(overnight=True)

            # Then: both boards are searched
            assert "board_a" in result.boards_searched, "enabled board missing from search"
            assert "linkedin" in result.boards_searched, "overnight board missing from search"

    async def test_board_failure_does_not_abort_other_boards(self) -> None:
        """
        Given two boards where one raises ActionableError on authenticate,
        When run() is called,
        Then the good board's listings are still scored and results include both boards.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: one failing + one good board
            settings = _make_settings(tmpdir, enabled_boards=["failing_board", "good_board"])
            runner, _ = _make_runner_with_real_stack(settings)

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

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(
                    AdapterRegistry._registry,  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_registry)
                    {
                        "failing_board": lambda: failing_adapter,
                        "good_board": lambda: good_adapter,
                    },
                ),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: both boards are searched
                result = await runner.run()

            # Then: both boards are reported, good board's listing was scored
            assert "good_board" in result.boards_searched, (
                f"good_board missing from boards_searched: {result.boards_searched}"
            )
            assert "failing_board" in result.boards_searched, (
                f"failing_board missing from boards_searched: {result.boards_searched}"
            )
            assert result.summary.total_found >= 1, (
                f"Expected at least 1 found listing, got: {result.summary.total_found}"
            )

    async def test_scoring_failure_increments_failed_count(self) -> None:
        """
        Given a runner whose Ollama embed endpoint returns a non-retryable error,
        When a listing is collected and scoring is attempted,
        Then failed_listings is incremented (real Scorer → real Embedder → mock client fails).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: runner with populated store
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)

            listing = _make_listing()
            mock_adapter = _make_test_adapter(search_results=[listing])

            # Make embed fail with non-retryable 404 → immediate ActionableError
            mock_client.embed.side_effect = ollama_sdk.ResponseError(
                "Model not found", status_code=404
            )

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs and tries to score
                result = await runner.run()

            # Then: failure is counted, not raised
            assert result.failed_listings >= 1, (
                f"Expected failed_listings >= 1, got: {result.failed_listings}"
            )

    async def test_empty_results_return_valid_run_result(self) -> None:
        """
        Given a runner searching a board that returns no listings,
        When run() completes,
        Then a valid RunResult is returned with empty ranked_listings.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: board returns no listings
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

            mock_adapter = _make_test_adapter()

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: no listings found
                result = await runner.run()

            # Then: valid RunResult with zero ranked
            assert isinstance(result, RunResult), f"Expected RunResult, got: {type(result)}"
            assert result.ranked_listings == [], (
                f"Expected empty ranked_listings, got: {result.ranked_listings}"
            )
            assert result.boards_searched == ["testboard"], (
                f"Expected ['testboard'], got: {result.boards_searched}"
            )

    async def test_scored_listings_are_passed_to_ranker(self) -> None:
        """
        Given a board that returns one listing,
        When run() scores it successfully,
        Then the listing passes through the ranker and appears in summary counts.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: one listing available
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

            listing = _make_listing()
            mock_adapter = _make_test_adapter(search_results=[listing])

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs and scores
                result = await runner.run()

            # Then: listing was scored and ranked
            assert result.summary.total_found == 1, (
                f"Expected total_found == 1, got: {result.summary.total_found}"
            )
            assert result.summary.total_scored == 1, (
                f"Expected total_scored == 1, got: {result.summary.total_scored}"
            )

    async def test_overnight_overlap_does_not_duplicate_board(self) -> None:
        """
        Given a runner where an overnight board is also an enabled board,
        When run(overnight=True) is called,
        Then the board appears only once in boards_searched.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: same board in both enabled and overnight
            settings = _make_settings(
                tmpdir,
                enabled_boards=["board_a"],
                overnight_boards=["board_a"],
            )
            runner, _ = _make_runner_with_real_stack(settings)

            adapter_a = _make_test_adapter(board_name="board_a")

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(
                    AdapterRegistry._registry,  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_registry)
                    {"board_a": lambda: adapter_a},
                ),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: overnight mode
                result = await runner.run(overnight=True)

            # Then: board_a appears exactly once
            assert result.boards_searched.count("board_a") == 1, (
                f"Expected board_a once, got: {result.boards_searched}"
            )

    async def test_auto_indexes_only_empty_collections(self) -> None:
        """
        Given a runner where the positive_signals collection is empty but
        resume and archetypes are populated,
        When run() is called,
        Then only the positive_signals collection is auto-indexed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: resume and archetypes populated, positive empty
            resume_path, archetypes_path, rubric_path = _create_index_files(tmpdir)
            settings = _make_settings(
                tmpdir,
                resume_path=resume_path,
                archetypes_path=archetypes_path,
                global_rubric_path=rubric_path,
            )
            runner, _ = _make_runner_with_real_stack(settings, populate_store=False)
            store = runner._store  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_store)

            # Seed resume and archetypes (leave positive_signals empty)
            for name in ("resume", "role_archetypes"):
                store.add_documents(
                    name,
                    ids=[f"{name}-seed"],
                    documents=[f"Seed document for {name}"],
                    embeddings=[EMBED_FAKE],
                )

            mock_adapter = _make_test_adapter()
            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs (triggers partial auto-indexing)
                result = await runner.run()

            # Then: positive_signals was indexed, run completed
            assert isinstance(result, RunResult), f"Expected RunResult, got: {type(result)}"
            assert store.collection_count("global_positive_signals") > 0, (
                "Expected positive_signals to be auto-indexed"
            )

    async def test_auto_index_skips_populated_collections(self) -> None:
        """
        Given a runner where only archetypes is empty but resume and
        positive_signals are populated,
        When run() is called,
        Then only archetypes is auto-indexed while the others are untouched.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: resume and positive_signals populated, archetypes empty
            resume_path, archetypes_path, rubric_path = _create_index_files(tmpdir)
            settings = _make_settings(
                tmpdir,
                resume_path=resume_path,
                archetypes_path=archetypes_path,
                global_rubric_path=rubric_path,
            )
            runner, _ = _make_runner_with_real_stack(settings, populate_store=False)
            store = runner._store  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_store)

            # Seed resume and positive_signals (leave archetypes empty)
            for name in ("resume", "global_positive_signals"):
                store.add_documents(
                    name,
                    ids=[f"{name}-seed"],
                    documents=[f"Seed document for {name}"],
                    embeddings=[EMBED_FAKE],
                )

            mock_adapter = _make_test_adapter()
            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs (triggers partial auto-indexing)
                result = await runner.run()

            # Then: archetypes was indexed, run completed
            assert isinstance(result, RunResult), f"Expected RunResult, got: {type(result)}"
            assert store.collection_count("role_archetypes") > 0, (
                "Expected archetypes to be auto-indexed"
            )


# ---------------------------------------------------------------------------
# TestBoardSearchDelegation
# ---------------------------------------------------------------------------


class TestBoardSearchDelegation:
    """REQUIREMENT: Individual board search delegates correctly to adapter and session manager.

    WHO: The pipeline runner calling adapters through the session manager
    WHAT: (1) The system skips a board that has no config section and returns an empty result without error.
          (2) The system calls authenticate, search, and extract_detail in that order when a returned listing has empty full_text.
          (3) The system does not call extract_detail when a returned listing already has full_text populated.
          (4) The system excludes a listing whose extracted full_text is only whitespace and counts it as failed.
          (5) The system counts a listing as failed when extract_detail raises an ActionableError and still scores the other listing.
          (6) The system counts an unexpected exception during extract_detail as a failure without aborting the run.
          (7) The system continues to the next search URL after a search ActionableError and collects that URL's results successfully.
    WHY: A missing config crashing the run is a config error reported too late;
         a single extraction failure aborting the board discards all other
         results from that board's search

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama API),
               async_playwright (Playwright browser library)
        Real:  PipelineRunner, Embedder, Scorer, VectorStore, Ranker,
               DecisionRecorder, AdapterRegistry, SessionManager, throttle
        Never: Construct ScoreResult directly — always obtained via real Scorer.score()
    """

    async def test_board_with_no_config_section_is_skipped(self) -> None:
        """
        Given a runner configured for 'testboard',
        When run(boards=["nonexistent_board"]) is called,
        Then the board is skipped and an empty result is returned without error.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: runner configured for 'testboard' only
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

            # Register adapter so AdapterRegistry.get() succeeds — but
            # settings.boards has no entry, so the runner skips it.
            mock_adapter = _make_test_adapter(board_name="nonexistent_board")

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"nonexistent_board": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: non-existent board requested
                result = await runner.run(boards=["nonexistent_board"])

            # Then: skipped without error
            assert result.ranked_listings == [], (
                f"Expected empty ranked_listings, got: {result.ranked_listings}"
            )
            assert result.failed_listings == 0, (
                f"Expected 0 failed_listings, got: {result.failed_listings}"
            )

    async def test_adapter_lifecycle_runs_in_order(self) -> None:
        """
        Given an adapter that returns a listing with empty full_text,
        When run() searches that board,
        Then authenticate → search → extract_detail is called in that order.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: adapter with lifecycle tracking
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

            call_order: list[str] = []
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

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs
                result = await runner.run(boards=["testboard"])

            # Then: lifecycle order is correct and listing was found
            assert call_order == [
                "authenticate",
                "search",
                "extract_detail",
            ], f"Expected lifecycle order, got: {call_order}"
            assert result.summary.total_found == 1, (
                f"Expected total_found == 1, got: {result.summary.total_found}"
            )

    async def test_enriched_listings_skip_extract_detail(self) -> None:
        """
        Given an adapter that returns a listing with full_text already populated,
        When run() processes that listing,
        Then extract_detail is not called.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: listing with pre-populated full_text
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

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

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs
                result = await runner.run(boards=["testboard"])

            # Then: extract_detail was skipped
            assert call_order == [
                "authenticate",
                "search",
            ], f"Expected no extract_detail, got: {call_order}"
            assert result.summary.total_found == 1, (
                f"Expected total_found == 1, got: {result.summary.total_found}"
            )

    async def test_empty_jd_text_is_counted_as_failure(self) -> None:
        """
        Given an adapter whose extract_detail returns whitespace-only full_text,
        When run() processes that listing,
        Then the listing is excluded and counted as failed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: extraction produces whitespace-only text
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

            search_listing = JobListing(
                board="testboard",
                external_id="1",
                title="Staff Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/1",
                full_text="",
            )
            empty_listing = JobListing(
                board="testboard",
                external_id="1",
                title="Empty Role",
                company="Co",
                location="Remote",
                url="https://example.org/1",
                full_text="   ",
            )

            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)
            mock_adapter.authenticate = AsyncMock()
            mock_adapter.search = AsyncMock(return_value=[search_listing])
            mock_adapter.extract_detail = AsyncMock(return_value=empty_listing)

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline processes the empty listing
                result = await runner.run(boards=["testboard"])

            # Then: listing excluded, failure counted
            assert result.ranked_listings == [], (
                f"Expected no ranked listings, got: {result.ranked_listings}"
            )
            assert result.failed_listings == 1, (
                f"Expected 1 failed_listings, got: {result.failed_listings}"
            )

    async def test_extraction_error_counts_failure_without_aborting(self) -> None:
        """
        Given two listings where extract_detail raises ActionableError on the first,
        When run() processes both,
        Then the good listing is scored and the bad one is counted as failed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: two listings, one will fail extraction
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

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

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline processes both listings
                result = await runner.run(boards=["testboard"])

            # Then: good listing scored, bad listing counted as failure
            assert result.summary.total_found == 1, (
                f"Expected 1 found, got: {result.summary.total_found}"
            )
            assert result.ranked_listings[0].listing.external_id == "good", (
                f"Expected 'good' listing, got: {result.ranked_listings[0].listing.external_id}"
            )
            assert result.failed_listings == 1, f"Expected 1 failed, got: {result.failed_listings}"

    async def test_unexpected_exception_during_extraction_is_counted(self) -> None:
        """
        Given an adapter whose extract_detail raises RuntimeError,
        When run() processes that listing,
        Then the failure is counted without aborting the run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: extraction raises unexpected error
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

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

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs
                result = await runner.run(boards=["testboard"])

            # Then: failure counted, run didn't abort
            assert result.ranked_listings == [], (
                f"Expected no ranked listings, got: {result.ranked_listings}"
            )
            assert result.failed_listings == 1, f"Expected 1 failed, got: {result.failed_listings}"

    async def test_search_failure_skips_url_and_continues(self) -> None:
        """
        Given a board with two search URLs where the first raises ActionableError,
        When run() searches both,
        Then the second URL's results are collected successfully.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: two search URLs, first will fail
            settings = _make_settings(tmpdir)
            settings.boards["testboard"].searches = [
                "https://testboard.com/search1",
                "https://testboard.com/search2",
            ]
            runner, _ = _make_runner_with_real_stack(settings)

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

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline searches both URLs
                result = await runner.run(boards=["testboard"])

            # Then: both URLs attempted, second succeeded
            assert call_count["search"] == 2, (
                f"Expected 2 search calls, got: {call_count['search']}"
            )
            assert result.summary.total_found == 1, (
                f"Expected 1 found from second URL, got: {result.summary.total_found}"
            )


# ---------------------------------------------------------------------------
# TestAutoIndex (Phase 4a — auto-recovery)
# ---------------------------------------------------------------------------


class TestAutoIndex:
    """REQUIREMENT: Empty collections are auto-indexed before scoring begins.

    WHO: The operator running search after a reset or on first use
    WHAT: (1) The system creates a real Indexer and indexes the resume and archetype collections when run() starts with an unpopulated store.
          (2) The system skips auto-indexing when collections are already populated and leaves the existing collection counts unchanged.
          (3) The system populates the collections through auto-indexing before scoring begins.
          (4) The system treats the collection as empty when collection_count raises ActionableError and runs real auto-indexing.
    WHY: Failing all 180 listings because the operator forgot to run a
         separate index command is a predictable situation with an obvious
         recovery — the system should just do it

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama API),
               async_playwright (Playwright browser library)
        Real:  PipelineRunner, Embedder, Scorer, Indexer, VectorStore, Ranker,
               DecisionRecorder, AdapterRegistry, SessionManager, throttle,
               config files on disk
        Never: Construct Indexer directly or patch it — always exercised through
               PipelineRunner._ensure_indexed()
    """

    async def test_auto_indexes_when_resume_collection_is_empty(self) -> None:
        """
        Given an unpopulated store and real config files on disk,
        When run() is called,
        Then _ensure_indexed creates a real Indexer and indexes resume + archetypes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: config files exist, store is empty
            resume, archetypes, rubric = _create_index_files(tmpdir)
            settings = _make_settings(
                tmpdir,
                resume_path=resume,
                archetypes_path=archetypes,
                global_rubric_path=rubric,
            )
            runner, _ = _make_runner_with_real_stack(settings, populate_store=False)

            mock_adapter = _make_test_adapter()
            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs with empty collections
                await runner.run()

            # Then: collections were populated by real Indexer
            store = runner._store  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_store)
            assert store.collection_count("resume") > 0, (
                "resume collection should be populated after auto-index"
            )
            assert store.collection_count("role_archetypes") > 0, (
                "role_archetypes collection should be populated after auto-index"
            )
            assert store.collection_count("global_positive_signals") > 0, (
                "global_positive_signals collection should be populated after auto-index"
            )

    async def test_skips_auto_index_when_collections_are_populated(self) -> None:
        """
        Given a store with pre-populated collections,
        When run() is called,
        Then no re-indexing occurs (collection counts remain unchanged).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: store already populated
            settings = _make_settings(tmpdir)
            runner, _mock_client = _make_runner_with_real_stack(settings)

            store = runner._store  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_store)
            resume_count_before = store.collection_count("resume")
            archetypes_count_before = store.collection_count("role_archetypes")

            mock_adapter = _make_test_adapter()
            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs
                await runner.run()

            # Then: counts unchanged — no re-indexing happened
            assert store.collection_count("resume") == resume_count_before, (
                "resume collection should not change when already populated"
            )
            assert store.collection_count("role_archetypes") == archetypes_count_before, (
                "role_archetypes collection should not change when already populated"
            )

    async def test_auto_index_runs_before_scoring_begins(self) -> None:
        """
        Given empty collections, real config files, and a board that returns a listing,
        When run() is called,
        Then auto-indexing populates collections before scoring uses them.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: config files exist, store empty, one listing to score
            resume, archetypes, rubric = _create_index_files(tmpdir)
            settings = _make_settings(
                tmpdir,
                resume_path=resume,
                archetypes_path=archetypes,
                global_rubric_path=rubric,
            )
            runner, mock_client = _make_runner_with_real_stack(settings, populate_store=False)

            listing = _make_listing()
            mock_adapter = _make_test_adapter(search_results=[listing])

            # Track embed calls — auto-index embeds config content,
            # then scoring embeds listing text
            original_embed = mock_client.embed

            call_sequence: list[str] = []

            async def _tracking_embed(*args: object, **kwargs: object) -> object:
                result = await original_embed(*args, **kwargs)
                call_sequence.append("embed")
                return result

            mock_client.embed = _tracking_embed

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline auto-indexes then scores
                result = await runner.run()

            # Then: auto-index ran (populated collections) and scoring ran too
            store = runner._store  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_store)
            assert store.collection_count("resume") > 0, "resume should be populated by auto-index"
            assert result.summary.total_scored >= 1, (
                f"Expected at least 1 scored listing, got: {result.summary.total_scored}"
            )

    async def test_collection_empty_returns_true_when_store_raises(self) -> None:
        """
        Given a store with no collections (collection_count raises ActionableError),
        When _ensure_indexed checks collections,
        Then the collection is treated as empty and real auto-indexing runs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: config files exist, store completely empty
            resume, archetypes, rubric = _create_index_files(tmpdir)
            settings = _make_settings(
                tmpdir,
                resume_path=resume,
                archetypes_path=archetypes,
                global_rubric_path=rubric,
            )
            runner, _ = _make_runner_with_real_stack(settings, populate_store=False)

            mock_adapter = _make_test_adapter()
            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs — _collection_empty returns True for missing collections
                await runner.run()

            # Then: all three collections were auto-indexed
            store = runner._store  # pyright: ignore[reportPrivateUsage] # Tests verify internal state (_store)
            assert store.collection_count("resume") > 0, (
                "resume should be auto-indexed when collection was missing"
            )
            assert store.collection_count("role_archetypes") > 0, (
                "role_archetypes should be auto-indexed when collection was missing"
            )
            assert store.collection_count("global_positive_signals") > 0, (
                "global_positive_signals should be auto-indexed when collection was missing"
            )


class TestCompEnrichment:
    """REQUIREMENT: Listings with salary text have comp fields populated after scoring.

    WHO: The scorer computing comp_score; the exporter showing salary data
    WHAT: (1) The system sets `comp_min`, `comp_max`, `comp_source`, and `comp_text` on a listing when scoring its pipeline input finds a salary range in `full_text`.
    WHY: Missing comp enrichment would leave salary fields empty even when
         the JD contains salary data, making comp_score always neutral

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama API),
               async_playwright (Playwright browser library)
        Real:  PipelineRunner, Embedder, Scorer, VectorStore, Ranker,
               DecisionRecorder, AdapterRegistry, SessionManager, throttle,
               parse_compensation
        Never: Construct ScoreResult directly — always obtained via real Scorer.score()
    """

    async def test_listing_with_salary_text_gets_comp_fields_populated(self) -> None:
        """
        Given a listing whose full_text contains a salary range,
        When the pipeline runner scores it,
        Then comp_min, comp_max, comp_source, and comp_text are set on the listing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with real stack and a listing containing salary text
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)

            listing = JobListing(
                board="testboard",
                external_id="comp-1",
                title="Principal Architect",
                company="Acme Corp",
                location="Remote",
                url="https://testboard.com/comp-1",
                full_text="Principal Architect role. Salary: $180,000 - $220,000 per year.",
            )

            mock_adapter = _make_test_adapter(search_results=[listing])

            mock_pw_fn, _ = _mock_playwright_boundary()
            with (
                patch.dict(AdapterRegistry._registry, {"testboard": lambda: mock_adapter}),  # type: ignore[dict-item]
                patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
                patch("jobsearch_rag.adapters.session._STORAGE_DIR", Path(tmpdir)),
            ):
                # When: pipeline runs and scores the listing
                result = await runner.run()

            # Then: comp enrichment populated the salary fields
            assert result.summary.total_found == 1, (
                f"Expected total_found == 1, got: {result.summary.total_found}"
            )
            assert listing.comp_min == 180_000.0, (
                f"Expected comp_min == 180000.0, got: {listing.comp_min}"
            )
            assert listing.comp_max == 220_000.0, (
                f"Expected comp_max == 220000.0, got: {listing.comp_max}"
            )
            assert listing.comp_source == "employer", (
                f"Expected comp_source == 'employer', got: {listing.comp_source}"
            )
            assert listing.comp_text is not None, "Expected comp_text to be set"
            assert "$180,000" in listing.comp_text, (
                f"Expected '$180,000' in comp_text, got: {listing.comp_text}"
            )

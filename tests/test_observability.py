"""
Observability tests — structured session tracing and log correlation.

Maps to BDD spec: TestSessionTracing
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

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
from jobsearch_rag.pipeline.runner import PipelineRunner
from tests.constants import EMBED_FAKE

if TYPE_CHECKING:
    from jobsearch_rag.rag.store import VectorStore

# ---------------------------------------------------------------------------
# Public API surface (from src/jobsearch_rag/logging):
#   new_session_id() -> str
#   configure_session_logging(log_dir: str, session_id: str, *, level: int) -> logging.FileHandler
#
# Public API surface (from src/jobsearch_rag/pipeline/runner):
#   PipelineRunner(settings: Settings) -> None
#   PipelineRunner.run(boards, *, overnight, force_rescore) -> RunResult
#   RunResult.ranked_listings, .summary, .failed_listings, .errors, .boards_searched
#
# Events emitted by structured logging:
#   "score_computed" — one per scored listing, includes component scores
#   "session_summary" — one per run, closes the session
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers — same patterns as test_runner.py
# ---------------------------------------------------------------------------


def _make_settings(
    tmpdir: str,
    enabled_boards: list[str] | None = None,
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
    return Settings(
        enabled_boards=boards,
        overnight_boards=[],
        boards=board_configs,
        scoring=ScoringConfig(disqualify_on_llm_flag=False),
        ollama=OllamaConfig(),
        output=OutputConfig(output_dir=str(Path(tmpdir) / "output")),
        chroma=ChromaConfig(persist_dir=str(Path(tmpdir) / "chroma_db")),
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
) -> tuple[PipelineRunner, AsyncMock]:
    """
    Create a PipelineRunner with real stack and mocked Ollama client.

    Only ``ollama_sdk.AsyncClient`` is mocked — the I/O boundary.
    """
    mock_client = AsyncMock()

    model_embed = MagicMock()
    model_embed.model = settings.ollama.embed_model
    model_llm = MagicMock()
    model_llm.model = settings.ollama.llm_model
    list_response = MagicMock()
    list_response.models = [model_embed, model_llm]
    mock_client.list.return_value = list_response

    embed_response = MagicMock()
    embed_response.embeddings = [EMBED_FAKE]
    mock_client.embed.return_value = embed_response

    with patch(
        "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        return_value=mock_client,
    ):
        runner = PipelineRunner(settings)

    _populate_store(runner._store)  # pyright: ignore[reportPrivateUsage]

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
    """Mock async_playwright — the Playwright I/O boundary."""
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
    """Create a test adapter with configurable search results."""
    adapter = MagicMock()
    adapter.board_name = board_name
    adapter.rate_limit_seconds = (0.0, 0.0)
    adapter.authenticate = AsyncMock()
    adapter.search = AsyncMock(return_value=search_results if search_results is not None else [])
    adapter.extract_detail = AsyncMock()
    return adapter


def _run_pipeline_and_read_logs(
    tmpdir: str,
    listings: list[JobListing],
    mock_client: AsyncMock,
    runner: PipelineRunner,
    *,
    exclude_files: set[str] | None = None,
) -> list[dict[str, object]]:
    """
    Run the pipeline against the given listings and return parsed log entries.

    Sets up the classify mock to return a non-disqualified response,
    patches Playwright and adapter boundaries, runs the pipeline,
    then reads and parses the JSON-lines log file.

    *exclude_files* is a set of filenames from prior runs to skip.
    """
    mock_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"disqualified": false}'),
    )

    mock_async_pw, _mock_page = _mock_playwright_boundary()
    adapter = _make_test_adapter(search_results=listings)

    log_dir = Path(tmpdir) / "logs"

    with (
        patch("jobsearch_rag.adapters.session.async_playwright", mock_async_pw),
        patch.dict(
            AdapterRegistry._registry,  # pyright: ignore[reportPrivateUsage]
            {"testboard": type(adapter)},
        ),
        patch(
            "jobsearch_rag.adapters.AdapterRegistry.get",
            return_value=adapter,
        ),
        patch("jobsearch_rag.adapters.session.asyncio.sleep", new_callable=AsyncMock),
    ):
        asyncio.run(runner.run())

    # Parse JSON-lines log files (only new ones)
    skip = exclude_files or set()
    entries: list[dict[str, object]] = []
    for log_file in sorted(log_dir.glob("*.jsonl")):
        if log_file.name in skip:
            continue
        for line in log_file.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries


# ---------------------------------------------------------------------------
# TestSessionTracing
# ---------------------------------------------------------------------------


class TestSessionTracing:
    """
    REQUIREMENT: Every run produces a structured log file whose entries
    are correlated by a session ID so the operator can reconstruct exactly
    what happened in any given run.

    WHO: The operator diagnosing unexpected scores, missed listings, or
         slow inference after a run completes
    WHAT: (1) A session ID is generated at the start of each CLI invocation
              and appears in every log entry written during that run
          (2) log entries are written in JSON-lines format
          (3) the session log is written to data/logs/ alongside stderr output
          (4) a run that processes three listings produces at least three
              score_computed log entries all sharing the same session ID
          (5) a second run produces a different session ID from the first
          (6) the session ID appears in the session_summary entry that
              closes every run
          (7) a score_computed entry contains all six component scores
    WHY: Mixed log output from concurrent runs or re-runs is unreadable
         without a correlation key

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama HTTP); async_playwright
               (Playwright I/O); asyncio.sleep for throttle bypass
        Real:  PipelineRunner, logging infrastructure, log file in tmp_path
        Never: Mock the logger or inject session IDs directly; run the
               real pipeline and verify the session ID by parsing the
               actual log file written to tmp_path
    """

    def test_every_log_entry_in_a_run_shares_the_same_session_id(self) -> None:
        """
        Given a pipeline run that processes two listings
        When the run completes and the log file is read
        Then every JSON-lines entry contains a 'session' field
        And all 'session' values in that file are identical
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner and two listings
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [
                _make_listing(external_id="1", title="Engineer A"),
                _make_listing(external_id="2", title="Engineer B"),
            ]

            # When: pipeline runs and log file is read
            entries = _run_pipeline_and_read_logs(tmpdir, listings, mock_client, runner)

            # Then: every entry has a 'session' field and all values match
            assert len(entries) > 0, "Expected at least one log entry"
            session_ids = {e["session"] for e in entries if "session" in e}
            assert len(session_ids) == 1, (
                f"Expected all entries to share one session ID, got {session_ids}"
            )
            for entry in entries:
                assert "session" in entry, f"Entry missing 'session' field: {entry}"

    def test_session_id_appears_in_session_summary_log_entry(self) -> None:
        """
        Given a completed pipeline run
        When the log file is read
        Then the entry whose 'event' is 'session_summary' contains
        the same session ID as all other entries in that file
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with one listing
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [_make_listing()]

            # When: pipeline runs
            entries = _run_pipeline_and_read_logs(tmpdir, listings, mock_client, runner)

            # Then: session_summary entry exists with matching session ID
            summaries = [e for e in entries if e.get("event") == "session_summary"]
            assert len(summaries) == 1, f"Expected 1 session_summary entry, got {len(summaries)}"
            all_sessions = {e["session"] for e in entries}
            assert len(all_sessions) == 1, f"Expected uniform session ID, got {all_sessions}"
            assert summaries[0]["session"] in all_sessions, (
                "session_summary session ID does not match other entries"
            )

    def test_two_consecutive_runs_produce_different_session_ids(self) -> None:
        """
        Given two sequential pipeline runs against the same input
        When both log files are read
        Then the session IDs in the two files are different
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with one listing, run twice
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [_make_listing()]

            # When: two runs produce two sets of entries
            entries_1 = _run_pipeline_and_read_logs(tmpdir, listings, mock_client, runner)
            # Track files from run 1 so run 2 only reads its own
            log_dir = Path(tmpdir) / "logs"
            run1_files = {f.name for f in log_dir.glob("*.jsonl")}
            entries_2 = _run_pipeline_and_read_logs(
                tmpdir,
                listings,
                mock_client,
                runner,
                exclude_files=run1_files,
            )

            # Then: session IDs differ
            sessions_1 = {e["session"] for e in entries_1}
            sessions_2 = {e["session"] for e in entries_2}
            assert len(sessions_1) == 1, f"Run 1 had multiple session IDs: {sessions_1}"
            assert len(sessions_2) == 1, f"Run 2 had multiple session IDs: {sessions_2}"
            assert sessions_1 != sessions_2, (
                f"Two runs should have different session IDs, both got {sessions_1}"
            )

    def test_score_computed_entries_appear_once_per_scored_listing(self) -> None:
        """
        Given a pipeline run that scores three listings
        When the log file is read
        Then exactly three entries with event 'score_computed' are present
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with three listings
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [
                _make_listing(external_id="1", title="Role A"),
                _make_listing(external_id="2", title="Role B"),
                _make_listing(external_id="3", title="Role C"),
            ]

            # When: pipeline runs
            entries = _run_pipeline_and_read_logs(tmpdir, listings, mock_client, runner)

            # Then: exactly three score_computed entries
            score_entries = [e for e in entries if e.get("event") == "score_computed"]
            assert len(score_entries) == 3, (
                f"Expected 3 score_computed entries, got {len(score_entries)}"
            )

    def test_score_computed_entry_contains_all_six_component_scores(self) -> None:
        """
        Given a completed pipeline run
        When a 'score_computed' log entry is read
        Then it contains numeric fields for archetype, fit, culture,
        history, negative, and final scores
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with one listing
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [_make_listing()]

            # When: pipeline runs
            entries = _run_pipeline_and_read_logs(tmpdir, listings, mock_client, runner)

            # Then: score_computed entry has all six numeric fields
            score_entries = [e for e in entries if e.get("event") == "score_computed"]
            assert len(score_entries) >= 1, "Expected at least one score_computed entry"
            entry = score_entries[0]
            expected_fields = ["archetype", "fit", "culture", "history", "negative", "final"]
            for field in expected_fields:
                assert field in entry, f"score_computed entry missing '{field}': {entry}"
                assert isinstance(entry[field], (int, float)), (
                    f"'{field}' should be numeric, got {type(entry[field])}: {entry[field]}"
                )

    def test_log_file_is_written_to_data_logs_directory(self) -> None:
        """
        Given a pipeline run using a configured data directory
        When the run completes
        Then a log file exists under data/logs/ within that directory
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with tmp data dir
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [_make_listing()]

            # When: pipeline runs
            _run_pipeline_and_read_logs(tmpdir, listings, mock_client, runner)

            # Then: log file exists under logs/
            log_dir = Path(tmpdir) / "logs"
            log_files = list(log_dir.glob("*.jsonl"))
            assert len(log_files) >= 1, (
                f"Expected at least one .jsonl log file in {log_dir}, found none"
            )

    def test_log_entries_are_valid_json_lines(self) -> None:
        """
        Given a completed pipeline run
        When each line of the log file is parsed as JSON
        Then every line parses without error
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: a runner with one listing
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [_make_listing()]

            # When: pipeline runs
            _run_pipeline_and_read_logs(tmpdir, listings, mock_client, runner)

            # Then: every line in every log file is valid JSON
            log_dir = Path(tmpdir) / "logs"
            log_files = list(log_dir.glob("*.jsonl"))
            assert len(log_files) >= 1, f"Expected at least one .jsonl log file in {log_dir}"
            for log_file in log_files:
                for i, line in enumerate(log_file.read_text().splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        msg = f"Line {i + 1} in {log_file.name} is not valid JSON: {exc}"
                        raise AssertionError(msg) from exc

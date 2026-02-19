"""CLI handler tests — parser construction, command wiring, output formatting.

Maps to BDD specs: TestParserConstruction, TestBoardsCommand, TestIndexCommand,
TestSearchCommand, TestDecideCommand, TestExportCommand
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.cli import (
    build_parser,
    handle_boards,
    handle_decide,
    handle_export,
    handle_index,
    handle_login,
    handle_reset,
    handle_review,
    handle_search,
)
from jobsearch_rag.config import (
    BoardConfig,
    ChromaConfig,
    OllamaConfig,
    OutputConfig,
    ScoringConfig,
    Settings,
)
from jobsearch_rag.pipeline.ranker import RankedListing, RankSummary
from jobsearch_rag.pipeline.runner import RunResult
from jobsearch_rag.rag.scorer import ScoreResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmpdir: str) -> Settings:
    """Create a valid Settings instance pointing at temp directories."""
    output_dir = str(Path(tmpdir) / "output")
    return Settings(
        enabled_boards=["testboard"],
        overnight_boards=[],
        boards={
            "testboard": BoardConfig(
                name="testboard",
                searches=["https://example.org/search"],
                max_pages=2,
                headless=True,
            ),
        },
        scoring=ScoringConfig(),
        ollama=OllamaConfig(),
        output=OutputConfig(output_dir=output_dir),
        chroma=ChromaConfig(persist_dir=tmpdir),
        resume_path="data/resume.md",
        archetypes_path="config/role_archetypes.toml",
    )


def _make_listing(
    board: str = "testboard",
    external_id: str = "1",
    title: str = "Staff Architect",
    company: str = "Acme Corp",
) -> JobListing:
    """Create a real JobListing with controlled values.

    Uses the actual dataclass rather than MagicMock so that optional
    fields (comp_min, comp_max, etc.) default to ``None`` instead of
    auto-generating child Mocks that serialize as ``<MagicMock …>``
    strings in CSV exports.
    """
    return JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company=company,
        location="Remote",
        url=f"https://example.org/{external_id}",
        full_text="A test job description.",
    )


def _make_ranked(
    final_score: float = 0.75,
    board: str = "testboard",
    external_id: str = "1",
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    duplicate_boards: list[str] | None = None,
) -> RankedListing:
    """Create a RankedListing with controlled values."""
    listing = _make_listing(board=board, external_id=external_id, title=title, company=company)
    scores = ScoreResult(
        fit_score=0.8,
        archetype_score=0.7,
        history_score=0.5,
        disqualified=False,
    )
    ranked = RankedListing(
        listing=listing,
        scores=scores,
        final_score=final_score,
    )
    if duplicate_boards:
        ranked.duplicate_boards = duplicate_boards
    return ranked


# ---------------------------------------------------------------------------
# TestParserConstruction
# ---------------------------------------------------------------------------


class TestParserConstruction:
    """REQUIREMENT: The CLI parser defines all subcommands with correct arguments.

    WHO: The operator invoking the tool from the command line
    WHAT: All five subcommands (index, search, decide, export, boards) are
          registered; each subcommand accepts its documented flags; a
          missing subcommand produces a usage error
    WHY: Silently ignoring a flag or failing on a valid subcommand breaks
         the operator's workflow before any real work begins
    """

    def test_parser_accepts_index_subcommand(self) -> None:
        """The 'index' subcommand is registered and parseable."""
        parser = build_parser()
        args = parser.parse_args(["index"])
        assert args.command == "index"

    def test_index_accepts_resume_only_flag(self) -> None:
        """The --resume-only flag on 'index' is accepted and defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["index"])
        assert args.resume_only is False
        args = parser.parse_args(["index", "--resume-only"])
        assert args.resume_only is True

    def test_parser_accepts_search_subcommand_with_all_flags(self) -> None:
        """The 'search' subcommand accepts --board, --overnight, and --open-top flags."""
        parser = build_parser()
        args = parser.parse_args(["search", "--board", "ziprecruiter", "--overnight", "--open-top", "5"])
        assert args.command == "search"
        assert args.board == "ziprecruiter"
        assert args.overnight is True
        assert args.open_top == 5

    def test_search_flags_default_to_none_and_false(self) -> None:
        """Search flags default to None/False when not provided."""
        parser = build_parser()
        args = parser.parse_args(["search"])
        assert args.board is None
        assert args.overnight is False
        assert args.open_top is None

    def test_parser_accepts_decide_subcommand_with_required_args(self) -> None:
        """The 'decide' subcommand requires job_id and --verdict."""
        parser = build_parser()
        args = parser.parse_args(["decide", "zr-123", "--verdict", "yes"])
        assert args.command == "decide"
        assert args.job_id == "zr-123"
        assert args.verdict == "yes"

    def test_decide_rejects_invalid_verdict(self) -> None:
        """An invalid --verdict value is rejected by the parser."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["decide", "zr-123", "--verdict", "invalid"])

    def test_parser_accepts_export_subcommand_with_format(self) -> None:
        """The 'export' subcommand accepts --format with choices."""
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "csv"])
        assert args.command == "export"
        assert args.format == "csv"

    def test_export_format_defaults_to_markdown(self) -> None:
        """The --format flag defaults to 'markdown' when not specified."""
        parser = build_parser()
        args = parser.parse_args(["export"])
        assert args.format == "markdown"

    def test_parser_accepts_boards_subcommand(self) -> None:
        """The 'boards' subcommand is registered and parseable."""
        parser = build_parser()
        args = parser.parse_args(["boards"])
        assert args.command == "boards"

    def test_parser_accepts_login_subcommand_with_board(self) -> None:
        """The 'login' subcommand requires --board and is parseable."""
        parser = build_parser()
        args = parser.parse_args(["login", "--board", "ziprecruiter"])
        assert args.command == "login"
        assert args.board == "ziprecruiter"

    def test_login_rejects_missing_board_flag(self) -> None:
        """The 'login' subcommand requires --board — omitting it is an error."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["login"])

    def test_missing_subcommand_raises_system_exit(self) -> None:
        """Omitting the subcommand entirely produces a usage error."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# TestBoardsCommand
# ---------------------------------------------------------------------------


class TestBoardsCommand:
    """REQUIREMENT: The boards command lists all registered adapters for operator discovery.

    WHO: The operator checking which boards are available before running a search
    WHAT: All registered board names are printed sorted alphabetically;
          an empty registry prints a clear 'no adapters' message;
          output uses a consistent bullet format
    WHY: An operator who can't see available boards guesses board names,
         leading to search failures on typos
    """

    def test_handle_boards_prints_sorted_board_names(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Registered boards are printed in sorted alphabetical order."""
        with patch("jobsearch_rag.cli.AdapterRegistry") as mock_registry:
            mock_registry.list_registered.return_value = ["ziprecruiter", "indeed", "linkedin"]
            handle_boards()
        output = capsys.readouterr().out
        assert "Registered adapters:" in output
        # Verify sorted order
        assert output.index("indeed") < output.index("linkedin") < output.index("ziprecruiter")

    def test_handle_boards_empty_registry_prints_no_adapters(self, capsys: pytest.CaptureFixture[str]) -> None:
        """An empty registry prints a clear message rather than blank output."""
        with patch("jobsearch_rag.cli.AdapterRegistry") as mock_registry:
            mock_registry.list_registered.return_value = []
            handle_boards()
        output = capsys.readouterr().out
        assert "No adapters registered." in output

    def test_handle_boards_uses_bullet_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Each board name is printed with a '  - ' bullet prefix."""
        with patch("jobsearch_rag.cli.AdapterRegistry") as mock_registry:
            mock_registry.list_registered.return_value = ["testboard"]
            handle_boards()
        output = capsys.readouterr().out
        assert "  - testboard" in output


# ---------------------------------------------------------------------------
# TestIndexCommand
# ---------------------------------------------------------------------------


class TestIndexCommand:
    """REQUIREMENT: The index command wires settings → embedder → indexer correctly.

    WHO: The operator running first-time setup or re-indexing after resume changes
    WHAT: Health check runs before any indexing; archetypes are indexed before
          resume by default; --resume-only skips archetypes; chunk counts are
          printed to stdout for operator sanity-check
    WHY: Indexing without a health check wastes time on a dead Ollama;
         silently skipping archetypes produces a broken scoring baseline
    """

    def test_index_runs_health_check_before_indexing(self) -> None:
        """Ollama health check runs before any indexing work begins."""
        mock_embedder = MagicMock()
        mock_embedder.health_check = AsyncMock()
        mock_indexer = MagicMock()
        mock_indexer.index_archetypes = AsyncMock(return_value=3)
        mock_indexer.index_resume = AsyncMock(return_value=5)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder", return_value=mock_embedder),
            patch("jobsearch_rag.rag.store.VectorStore"),
            patch("jobsearch_rag.rag.indexer.Indexer", return_value=mock_indexer),
        ):
            args = argparse.Namespace(resume_only=False)
            handle_index(args)

        mock_embedder.health_check.assert_awaited_once()

    def test_index_indexes_archetypes_and_resume_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without --resume-only, both archetypes and resume are indexed."""
        mock_embedder = MagicMock()
        mock_embedder.health_check = AsyncMock()
        mock_indexer = MagicMock()
        mock_indexer.index_archetypes = AsyncMock(return_value=3)
        mock_indexer.index_resume = AsyncMock(return_value=5)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder", return_value=mock_embedder),
            patch("jobsearch_rag.rag.store.VectorStore"),
            patch("jobsearch_rag.rag.indexer.Indexer", return_value=mock_indexer),
        ):
            args = argparse.Namespace(resume_only=False)
            handle_index(args)

        mock_indexer.index_archetypes.assert_awaited_once()
        mock_indexer.index_resume.assert_awaited_once()
        output = capsys.readouterr().out
        assert "Indexed 3 archetypes" in output
        assert "Indexed 5 resume chunks" in output

    def test_index_resume_only_skips_archetypes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With --resume-only, archetypes are not indexed."""
        mock_embedder = MagicMock()
        mock_embedder.health_check = AsyncMock()
        mock_indexer = MagicMock()
        mock_indexer.index_archetypes = AsyncMock(return_value=3)
        mock_indexer.index_resume = AsyncMock(return_value=5)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder", return_value=mock_embedder),
            patch("jobsearch_rag.rag.store.VectorStore"),
            patch("jobsearch_rag.rag.indexer.Indexer", return_value=mock_indexer),
        ):
            args = argparse.Namespace(resume_only=True)
            handle_index(args)

        mock_indexer.index_archetypes.assert_not_awaited()
        mock_indexer.index_resume.assert_awaited_once()
        output = capsys.readouterr().out
        assert "archetypes" not in output.lower()
        assert "Indexed 5 resume chunks" in output

    def test_index_prints_chunk_counts_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Index command reports archetype and resume chunk counts for operator sanity-check."""
        mock_embedder = MagicMock()
        mock_embedder.health_check = AsyncMock()
        mock_indexer = MagicMock()
        mock_indexer.index_archetypes = AsyncMock(return_value=7)
        mock_indexer.index_resume = AsyncMock(return_value=4)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder", return_value=mock_embedder),
            patch("jobsearch_rag.rag.store.VectorStore"),
            patch("jobsearch_rag.rag.indexer.Indexer", return_value=mock_indexer),
        ):
            args = argparse.Namespace(resume_only=False)
            handle_index(args)

        output = capsys.readouterr().out
        assert "7" in output
        assert "4" in output


# ---------------------------------------------------------------------------
# TestSearchCommand
# ---------------------------------------------------------------------------


class TestSearchCommand:
    """REQUIREMENT: The search command prints a structured summary and ranked listings.

    WHO: The operator reviewing search results in the terminal
    WHAT: The summary includes boards searched, total found, scored, deduplicated,
          excluded, failed, and final count; each ranked listing shows score,
          title, company, board, URL, and score explanation; duplicate boards
          are noted; --open-top opens browser tabs for the top N results
    WHY: Missing summary fields leave the operator guessing whether the run
         was healthy; missing listing fields prevent informed review
    """

    def test_search_prints_summary_with_all_required_fields(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The summary section includes all pipeline statistics."""
        result = RunResult(
            ranked_listings=[],
            summary=RankSummary(total_found=20, total_scored=18, total_deduplicated=3, total_excluded=2),
            failed_listings=2,
            boards_searched=["ziprecruiter"],
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=result)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
        ):
            args = argparse.Namespace(board=None, overnight=False, open_top=None, force_rescore=False)
            handle_search(args)

        output = capsys.readouterr().out
        assert "Boards searched:" in output
        assert "ziprecruiter" in output
        assert "Total found:" in output
        assert "20" in output
        assert "Scored:" in output
        assert "18" in output
        assert "Deduplicated:" in output
        assert "Failed:" in output
        assert "Final results:" in output

    def test_search_prints_ranked_listings_with_score_and_title(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each ranked listing shows its score, title, company, and URL."""
        ranked = _make_ranked(final_score=0.82, title="Staff Architect", company="Acme Corp")
        result = RunResult(
            ranked_listings=[ranked],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=result)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
            patch("webbrowser.open"),
        ):
            args = argparse.Namespace(board=None, overnight=False, open_top=None, force_rescore=False)
            handle_search(args)

        output = capsys.readouterr().out
        assert "0.82" in output
        assert "Staff Architect" in output
        assert "Acme Corp" in output

    def test_search_notes_duplicate_boards_on_listing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When a listing was seen on multiple boards, the duplicates are noted."""
        ranked = _make_ranked(duplicate_boards=["indeed", "linkedin"])
        result = RunResult(
            ranked_listings=[ranked],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=result)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
            patch("webbrowser.open"),
        ):
            args = argparse.Namespace(board=None, overnight=False, open_top=None, force_rescore=False)
            handle_search(args)

        output = capsys.readouterr().out
        assert "Also on:" in output
        assert "indeed" in output
        assert "linkedin" in output

    def test_search_board_flag_restricts_to_single_board(self) -> None:
        """The --board flag passes only that board name to the runner."""
        result = RunResult(
            ranked_listings=[],
            summary=RankSummary(),
            boards_searched=["indeed"],
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=result)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
        ):
            args = argparse.Namespace(board="indeed", overnight=False, open_top=None, force_rescore=False)
            handle_search(args)

        # Verify the runner was called with boards=["indeed"]
        mock_runner.run.assert_awaited_once()
        call_kwargs = mock_runner.run.call_args
        assert call_kwargs.kwargs.get("boards") or call_kwargs.args[0] == ["indeed"]

    def test_search_open_top_opens_browser_tabs(self) -> None:
        """The --open-top flag opens the top N results in the browser."""
        ranked = _make_ranked(final_score=0.9)
        result = RunResult(
            ranked_listings=[ranked],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=result)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
            patch("webbrowser.open") as mock_open,
        ):
            args = argparse.Namespace(board=None, overnight=False, open_top=1, force_rescore=False)
            handle_search(args)

        mock_open.assert_called_once()

    def test_search_no_open_top_and_settings_zero_opens_no_tabs(self) -> None:
        """When --open-top is not set and settings.open_top_n is 0, no tabs open."""
        ranked = _make_ranked(final_score=0.9)
        result = RunResult(
            ranked_listings=[ranked],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=result)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "jobsearch_rag.config.load_settings",
                return_value=Settings(
                    enabled_boards=["testboard"],
                    overnight_boards=[],
                    boards={},
                    scoring=ScoringConfig(),
                    ollama=OllamaConfig(),
                    output=OutputConfig(open_top_n=0, output_dir=str(Path(tmpdir) / "output")),
                    chroma=ChromaConfig(persist_dir=tmpdir),
                ),
            ),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
            patch("webbrowser.open") as mock_open,
        ):
            args = argparse.Namespace(board=None, overnight=False, open_top=None, force_rescore=False)
            handle_search(args)

        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# TestDecideCommand
# ---------------------------------------------------------------------------


class TestDecideCommand:
    """REQUIREMENT: The decide command records verdicts with appropriate error handling.

    WHO: The operator recording their assessment of a scored role
    WHAT: An existing job ID looks up and re-records with the new verdict;
          an unknown job ID exits with a clear error message;
          the recorded verdict and history count are printed to stdout
    WHY: Recording a verdict for a non-existent job silently corrupts the
         decision history; the operator needs confirmation the action took effect
    """

    def test_unknown_job_id_exits_with_error_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unknown job ID exits with a clear error rather than crashing."""
        mock_recorder = MagicMock()
        mock_recorder.get_decision.return_value = None

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder"),
            patch("jobsearch_rag.rag.store.VectorStore"),
            patch("jobsearch_rag.rag.decisions.DecisionRecorder", return_value=mock_recorder),
            pytest.raises(SystemExit) as exc_info,
        ):
            args = argparse.Namespace(job_id="nonexistent", verdict="yes", reason="")
            handle_decide(args)

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "No job found" in output

    def test_existing_job_records_verdict_and_prints_confirmation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Re-recording a verdict on a known job prints confirmation with history count."""
        mock_recorder = MagicMock()
        mock_recorder.get_decision.return_value = {
            "verdict": "maybe",
            "board": "ziprecruiter",
            "title": "Staff Architect",
            "company": "Acme",
        }
        mock_recorder.record = AsyncMock()
        mock_recorder.history_count.return_value = 5

        mock_store = MagicMock()
        mock_store.get_documents.return_value = {
            "documents": ["Full JD text here"],
            "ids": ["decision-zr-123"],
            "metadatas": [{}],
        }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder"),
            patch("jobsearch_rag.rag.store.VectorStore", return_value=mock_store),
            patch("jobsearch_rag.rag.decisions.DecisionRecorder", return_value=mock_recorder),
        ):
            args = argparse.Namespace(job_id="zr-123", verdict="yes", reason="")
            handle_decide(args)

        mock_recorder.record.assert_awaited_once()
        output = capsys.readouterr().out
        assert "Recorded 'yes' for zr-123" in output
        assert "5" in output

    def test_decide_with_reason_prints_reason_in_confirmation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When --reason is provided, confirmation output includes the reason text."""
        mock_recorder = MagicMock()
        mock_recorder.get_decision.return_value = {
            "verdict": "maybe",
            "board": "ziprecruiter",
            "title": "Staff Architect",
            "company": "Acme",
        }
        mock_recorder.record = AsyncMock()
        mock_recorder.history_count.return_value = 3

        mock_store = MagicMock()
        mock_store.get_documents.return_value = {
            "documents": ["Full JD text here"],
            "ids": ["decision-zr-123"],
            "metadatas": [{}],
        }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder"),
            patch("jobsearch_rag.rag.store.VectorStore", return_value=mock_store),
            patch("jobsearch_rag.rag.decisions.DecisionRecorder", return_value=mock_recorder),
        ):
            args = argparse.Namespace(
                job_id="zr-123", verdict="no", reason="Role requires on-call rotation"
            )
            handle_decide(args)

        call_kwargs = mock_recorder.record.call_args.kwargs
        assert call_kwargs["reason"] == "Role requires on-call rotation"
        output = capsys.readouterr().out
        assert "Role requires on-call rotation" in output

    def test_missing_jd_text_exits_with_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """If the JD text cannot be retrieved for a known job, exits with error."""
        mock_recorder = MagicMock()
        mock_recorder.get_decision.return_value = {
            "verdict": "maybe",
            "board": "ziprecruiter",
            "title": "Staff Architect",
            "company": "Acme",
        }

        mock_store = MagicMock()
        mock_store.get_documents.return_value = {
            "documents": [None],
            "ids": ["decision-zr-123"],
            "metadatas": [{}],
        }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.embedder.Embedder"),
            patch("jobsearch_rag.rag.store.VectorStore", return_value=mock_store),
            patch("jobsearch_rag.rag.decisions.DecisionRecorder", return_value=mock_recorder),
            pytest.raises(SystemExit) as exc_info,
        ):
            args = argparse.Namespace(job_id="zr-123", verdict="yes", reason="")
            handle_decide(args)

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "Could not retrieve JD text" in output


# ---------------------------------------------------------------------------
# TestExportCommand
# ---------------------------------------------------------------------------


class TestExportCommand:
    """REQUIREMENT: The export command re-exports saved results.

    WHO: The operator re-viewing results after a previous search run
    WHAT: The export command prints saved results in the requested format;
          when no results exist, prints a clear message and exits with code 1;
          each valid format is accepted
    WHY: Silently producing no output would leave the operator wondering
         if the command failed or if there were no results
    """

    def test_export_prints_format_and_stub_message(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Export prints saved markdown results when format is markdown."""
        md_file = tmp_path / "results.md"
        md_file.write_text("# Run Summary\n\nTest results.\n")

        with patch(
            "jobsearch_rag.config.load_settings",
            return_value=_make_settings(str(tmp_path / "chroma")),
        ) as mock_load:
            # Override output_dir to point to tmp_path
            mock_load.return_value.output = OutputConfig(output_dir=str(tmp_path))
            args = argparse.Namespace(format="markdown")
            handle_export(args)

        output = capsys.readouterr().out
        assert "# Run Summary" in output
        assert "Test results." in output

    def test_export_accepts_all_format_choices(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Each valid format is accepted; csv format prints CSV content."""
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("title,company\nStaff Architect,Acme\n")
        md_file = tmp_path / "results.md"
        md_file.write_text("# Results\n")

        with patch(
            "jobsearch_rag.config.load_settings",
            return_value=_make_settings(str(tmp_path / "chroma")),
        ) as mock_load:
            mock_load.return_value.output = OutputConfig(output_dir=str(tmp_path))

            args = argparse.Namespace(format="csv")
            handle_export(args)
            output = capsys.readouterr().out
            assert "Staff Architect" in output

            args = argparse.Namespace(format="markdown")
            handle_export(args)
            output = capsys.readouterr().out
            assert "# Results" in output

    def test_export_no_results_exits_with_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """When no previous results exist, exits with code 1."""
        with (
            patch(
                "jobsearch_rag.config.load_settings",
                return_value=_make_settings(str(tmp_path / "chroma")),
            ) as mock_load,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_load.return_value.output = OutputConfig(output_dir=str(tmp_path))
            args = argparse.Namespace(format="markdown")
            handle_export(args)

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "No previous results found" in output


# ---------------------------------------------------------------------------
# TestLoginCommand
# ---------------------------------------------------------------------------


class TestLoginCommand:
    """REQUIREMENT: The login command opens a headed browser for interactive authentication.

    WHO: The operator establishing a session before headless search runs
    WHAT: A headed (visible) browser opens to the board's login page;
          the operator completes login manually; session cookies are saved
          to ``data/{board}_session.json`` for reuse; the operator is
          prompted to press Enter when finished
    WHY: Cloudflare bot protection blocks headless browsers — logging in
         interactively establishes cookies that may enable headless operation
    """

    def test_login_opens_headed_browser_and_saves_session(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Login opens a headed browser, navigates to login URL, and saves session."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()

        mock_session = MagicMock()
        mock_session.new_page = AsyncMock(return_value=mock_page)
        mock_session.save_storage_state = AsyncMock(return_value="data/ziprecruiter_session.json")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "jobsearch_rag.adapters.session.SessionManager",
                return_value=mock_session,
            ) as mock_sm_cls,
            patch("builtins.input", return_value=""),
        ):
            args = argparse.Namespace(board="ziprecruiter", browser=None)
            handle_login(args)

        # Verify SessionManager was created with headless=False
        config = mock_sm_cls.call_args[0][0]
        assert config.headless is False
        assert config.board_name == "ziprecruiter"

        # Verify navigation to login URL
        mock_page.goto.assert_awaited_once()
        url_arg = mock_page.goto.call_args[0][0]
        assert "authn/login" in url_arg

        # Verify session was saved
        mock_session.save_storage_state.assert_awaited_once()

        output = capsys.readouterr().out
        assert "Session saved" in output

    def test_login_uses_board_specific_login_url(self) -> None:
        """Each board uses its own login URL."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()

        mock_session = MagicMock()
        mock_session.new_page = AsyncMock(return_value=mock_page)
        mock_session.save_storage_state = AsyncMock(return_value="data/linkedin_session.json")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("jobsearch_rag.adapters.session.SessionManager", return_value=mock_session),
            patch("builtins.input", return_value=""),
        ):
            args = argparse.Namespace(board="linkedin", browser=None)
            handle_login(args)

        url_arg = mock_page.goto.call_args[0][0]
        assert "linkedin.com/login" in url_arg

    def test_login_prints_instructions_for_operator(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Login prints clear instructions about what the operator needs to do."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()

        mock_session = MagicMock()
        mock_session.new_page = AsyncMock(return_value=mock_page)
        mock_session.save_storage_state = AsyncMock(return_value="data/ziprecruiter_session.json")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("jobsearch_rag.adapters.session.SessionManager", return_value=mock_session),
            patch("builtins.input", return_value=""),
        ):
            args = argparse.Namespace(board="ziprecruiter", browser=None)
            handle_login(args)

        output = capsys.readouterr().out
        assert "Interactive Login" in output
        assert "ziprecruiter" in output
        assert "Complete login" in output

    def test_login_browser_flag_sets_channel(self) -> None:
        """The --browser flag overrides the browser channel for the session."""
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()

        mock_session = MagicMock()
        mock_session.new_page = AsyncMock(return_value=mock_page)
        mock_session.save_storage_state = AsyncMock(return_value="data/ziprecruiter_session.json")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "jobsearch_rag.adapters.session.SessionManager",
                return_value=mock_session,
            ) as mock_sm_cls,
            patch("builtins.input", return_value=""),
        ):
            args = argparse.Namespace(board="ziprecruiter", browser="msedge")
            handle_login(args)

        config = mock_sm_cls.call_args[0][0]
        assert config.browser_channel == "msedge"


# ---------------------------------------------------------------------------
# TestSearchBrowserFailure
# ---------------------------------------------------------------------------


class TestSearchBrowserFailure:
    """REQUIREMENT: Browser open failures are reported gracefully, not as crashes.

    WHO: The operator running a search where webbrowser.open fails
    WHAT: When webbrowser.open raises an exception, the failure is printed
          but the search completes normally
    WHY: A browser failure should not discard valid search results
    """

    def test_webbrowser_open_failure_prints_error_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """GIVEN a search with --open-top that triggers a browser exception
        WHEN webbrowser.open raises
        THEN the error is printed and the search completes.
        """
        ranked = _make_ranked(final_score=0.9, external_id="fail-1")
        result = RunResult(
            ranked_listings=[ranked],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=result)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
            patch("webbrowser.open", side_effect=OSError("no browser")),
        ):
            args = argparse.Namespace(board=None, overnight=False, open_top=1, force_rescore=False)
            handle_search(args)

        output = capsys.readouterr().out
        assert "Failed to open" in output
        assert "no browser" in output


# ---------------------------------------------------------------------------
# TestExportMissing
# ---------------------------------------------------------------------------


class TestExportMissing:
    """REQUIREMENT: Requesting an export format with no file prints a helpful message.

    WHO: The operator running 'export' before any search has been done
    WHAT: When the requested format's file doesn't exist, a helpful
          message is printed instead of crashing
    WHY: Crashing on a missing file is unhelpful; the operator needs to
         know to run 'search' first
    """

    def test_export_format_not_found_prints_helpful_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """GIVEN results exist as markdown but not csv
        WHEN export --format csv is run
        THEN a message explains the format was not found.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            out_dir = Path(settings.output.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Create only the markdown file, not csv
            (out_dir / "results.md").write_text("# Results")

            with patch("jobsearch_rag.config.load_settings", return_value=settings):
                args = argparse.Namespace(format="csv")
                handle_export(args)

        output = capsys.readouterr().out
        assert "No csv export found" in output


# ---------------------------------------------------------------------------
# TestResetCommand
# ---------------------------------------------------------------------------


class TestResetCommand:
    """REQUIREMENT: The reset command clears ChromaDB collections and optionally output files.

    WHO: The operator starting a fresh run
    WHAT: 'reset' clears all known collections by default or a specific
          one via --collection; --clear-output removes the output directory
    WHY: Stale data from a previous run can corrupt scoring or mislead
         the operator into reviewing outdated results
    """

    def test_reset_all_collections_clears_all_known_collections(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """GIVEN no --collection flag
        WHEN handle_reset is run
        THEN all known collections are reset.
        """
        mock_store = MagicMock()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.store.VectorStore", return_value=mock_store),
        ):
            args = argparse.Namespace(collection=None, clear_output=False)
            handle_reset(args)

        output = capsys.readouterr().out
        assert "Reset complete" in output
        assert mock_store.reset_collection.call_count > 0

    def test_reset_single_collection(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """GIVEN --collection=resume
        WHEN handle_reset is run
        THEN only the 'resume' collection is reset.
        """
        mock_store = MagicMock()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("jobsearch_rag.config.load_settings", return_value=_make_settings(tmpdir)),
            patch("jobsearch_rag.rag.store.VectorStore", return_value=mock_store),
        ):
            args = argparse.Namespace(collection="resume", clear_output=False)
            handle_reset(args)

        output = capsys.readouterr().out
        assert "Reset collection: resume" in output
        mock_store.reset_collection.assert_called_once_with("resume")

    def test_reset_with_clear_output_removes_output_dir(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """GIVEN --clear-output flag
        WHEN handle_reset is run
        THEN the output directory is removed and recreated.
        """
        mock_store = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            out_dir = Path(settings.output.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "results.md").write_text("old data")

            with (
                patch("jobsearch_rag.config.load_settings", return_value=settings),
                patch("jobsearch_rag.rag.store.VectorStore", return_value=mock_store),
            ):
                args = argparse.Namespace(collection=None, clear_output=True)
                handle_reset(args)

        output = capsys.readouterr().out
        assert "Cleared output directory" in output


# ---------------------------------------------------------------------------
# TestReviewJdLoading
# ---------------------------------------------------------------------------


class TestReviewJdLoading:
    """
    REQUIREMENT: The review command populates each listing's full_text
    from JD files on disk, since CSV export does not store full text.

    WHO: The review CLI handler reconstructing RankedListing objects
         from CSV rows for the interactive review session
    WHAT: Each listing's full_text is loaded from the corresponding
          JD markdown file in output/jds/ using the rank-based filename
          convention ({rank:03d}_{company_slug}_{title_slug}.md); the
          JD body is extracted from the section after the
          '## Job Description' marker; missing JD files or files without
          the marker yield empty full_text (no crash); the slugify
          function normalizes company and title to lowercase hyphenated
          ASCII truncated at 80 characters
    WHY: DecisionRecorder requires full_text to generate embeddings for
         the history signal. Without it, recording a verdict fails with
         an empty-text validation error on the second listing reviewed
    """

    def test_review_populates_full_text_from_jd_file(self, tmp_path: Path) -> None:
        """
        When a CSV row has a matching JD file with a Job Description section
        Then the listing's full_text contains the JD body text
        """
        from jobsearch_rag.cli import _read_jd_text, _slugify

        # Given: A JD file with the expected name and content
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()
        slug_company = _slugify("Acme Corp")
        slug_title = _slugify("Staff Architect")
        filename = f"001_{slug_company}_{slug_title}.md"
        jd_content = (
            "# Staff Architect — Acme Corp\n\n"
            "**Score:** 0.85\n\n"
            "## Job Description\n"
            "We are looking for a Staff Architect to lead our platform team."
        )
        (jd_dir / filename).write_text(jd_content)

        # When: _read_jd_text is called with matching rank, title, company
        result = _read_jd_text(1, "Staff Architect", "Acme Corp", jd_dir=jd_dir)

        # Then: full_text contains the JD body
        assert result == "We are looking for a Staff Architect to lead our platform team.", (
            f"Expected JD body text, got: {result!r}"
        )

    def test_review_missing_jd_file_yields_empty_full_text(self, tmp_path: Path) -> None:
        """
        When the JD file does not exist on disk
        Then full_text is an empty string and no exception is raised
        """
        from jobsearch_rag.cli import _read_jd_text

        # Given: An empty jds directory (no matching file)
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()

        # When: _read_jd_text is called for a nonexistent file
        result = _read_jd_text(99, "Nonexistent Role", "Ghost Inc", jd_dir=jd_dir)

        # Then: Returns empty string
        assert result == "", (
            f"Expected empty string for missing JD file, got: {result!r}"
        )

    def test_review_jd_file_without_marker_yields_empty_full_text(
        self, tmp_path: Path
    ) -> None:
        """
        When the JD file exists but lacks the '## Job Description' marker
        Then full_text is an empty string
        """
        from jobsearch_rag.cli import _read_jd_text, _slugify

        # Given: A JD file that has no Job Description section
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()
        slug_company = _slugify("Acme Corp")
        slug_title = _slugify("Staff Architect")
        filename = f"001_{slug_company}_{slug_title}.md"
        (jd_dir / filename).write_text("# Staff Architect\n\nSome header text only.")

        # When: _read_jd_text is called
        result = _read_jd_text(1, "Staff Architect", "Acme Corp", jd_dir=jd_dir)

        # Then: Returns empty string since marker is absent
        assert result == "", (
            f"Expected empty string when marker is absent, got: {result!r}"
        )

    def test_review_jd_filename_matches_rank_company_title_slug(
        self, tmp_path: Path
    ) -> None:
        """
        When constructing the JD filename
        Then it follows the pattern {rank:03d}_{company_slug}_{title_slug}.md
        matching the export convention used by the search command
        """
        from jobsearch_rag.cli import _read_jd_text, _slugify

        # Given: A JD file named with the expected slug convention
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()
        # Company "O'Reilly Media" → "oreilly-media"
        # Title "Sr. Data Engineer (Remote)" → "sr-data-engineer-remote"
        company = "O'Reilly Media"
        title = "Sr. Data Engineer (Remote)"
        expected_name = f"005_{_slugify(company)}_{_slugify(title)}.md"
        jd_content = "## Job Description\nBuild data pipelines at scale."
        (jd_dir / expected_name).write_text(jd_content)

        # When: _read_jd_text is called with the original (unslugged) values
        result = _read_jd_text(5, title, company, jd_dir=jd_dir)

        # Then: The text is found because the filename slug matched
        assert result == "Build data pipelines at scale.", (
            f"Filename slug mismatch — could not load JD. Got: {result!r}"
        )

    def test_slugify_strips_special_chars_and_lowercases(self) -> None:
        """
        When slugifying company or title text
        Then special characters are stripped, spaces become hyphens,
        text is lowercased, and result is truncated to 80 characters
        """
        from jobsearch_rag.cli import _slugify

        # Given: Text with mixed case, special chars, and spaces
        assert _slugify("Acme Corp") == "acme-corp", "Basic two-word slug"
        assert _slugify("O'Reilly Media") == "oreilly-media", "Apostrophe stripped"
        assert _slugify("Sr. Data Engineer (Remote)") == "sr-data-engineer-remote", (
            "Dots and parens stripped"
        )
        assert _slugify("  Extra   Spaces  ") == "extra-spaces", (
            "Leading/trailing/multiple spaces collapsed"
        )
        # Truncation at 80 chars
        long_text = "a " * 50  # 100 chars before slugify
        assert len(_slugify(long_text)) <= 80, "Slug must be truncated to 80 chars"

    def test_open_listing_resolves_jd_file_via_slug_convention(
        self, tmp_path: Path
    ) -> None:
        """
        When the operator presses 'o' to open a JD
        Then the review session finds the JD file using the
        rank/company/title slug convention, not external_id
        """
        from unittest.mock import patch as mock_patch

        from jobsearch_rag.cli import _slugify
        from jobsearch_rag.pipeline.review import ReviewSession

        # Given: A JD file named with the slug convention at rank 3
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()
        company = "Acme Corp"
        title = "Staff Architect"
        slug_name = f"003_{_slugify(company)}_{_slugify(title)}.md"
        (jd_dir / slug_name).write_text("## Job Description\nFull JD here.")

        ranked = _make_ranked(title=title, company=company, external_id="zr-42")
        recorder = MagicMock()
        recorder.get_decision = MagicMock(return_value=None)
        session = ReviewSession(
            ranked_listings=[ranked],
            recorder=recorder,
            jd_dir=str(jd_dir),
        )

        # When: open_listing is called with the rank
        with mock_patch("jobsearch_rag.pipeline.review.webbrowser.open") as mock_open:
            session.open_listing(ranked, rank=3)
            mock_open.assert_called_once()
            opened_path = mock_open.call_args[0][0]

        # Then: It opened the slug-based JD file, not external_id.md
        assert slug_name in opened_path, (
            f"Expected slug-based filename '{slug_name}' in opened path, "
            f"got: {opened_path}"
        )


# ---------------------------------------------------------------------------
# TestReviewCommandHandler
# ---------------------------------------------------------------------------


class TestReviewCommandHandler:
    """
    REQUIREMENT: The review CLI handler wires dependencies, loads CSV,
    and drives the interactive loop so every operator action produces
    the correct output and side-effect.

    WHO: The operator running `python -m jobsearch_rag review`
    WHAT: Missing CSV prints 'No results found' and exits; CSV rows are
          faithfully reconstructed as RankedListings with all fields;
          all-decided state prints 'nothing to review'; the undecided
          count and command help are printed before the first listing;
          'q' stops review and reports how many were reviewed; 's' skips
          to the next listing without recording; 'y'/'n'/'m' records
          and prints confirmation (with reason when provided); 'o'
          delegates to session.open_listing with rank; invalid input
          reprints the command list; EOF/Ctrl-C is treated as quit;
          reviewing all listings prints 'Review complete'
    WHY: The handler is the orchestration boundary between user input
         and domain logic — untested wiring means the operator gets
         silent failures, missing output, or wired-wrong dependencies
         that only surface in production
    """

    _CSV_FIELDS = [
        "title", "company", "board", "location", "url",
        "fit_score", "archetype_score", "history_score",
        "comp_score", "final_score", "comp_min", "comp_max",
        "disqualified", "disqualifier_reason",
    ]

    @staticmethod
    def _csv_row(**overrides: str) -> dict[str, str]:
        """Build a CSV row dict with sensible defaults."""
        row: dict[str, str] = {
            "title": "Staff Architect",
            "company": "Acme Corp",
            "board": "ziprecruiter",
            "location": "Remote",
            "url": "https://example.org/job-1",
            "fit_score": "0.85",
            "archetype_score": "0.90",
            "history_score": "0.50",
            "comp_score": "0.60",
            "final_score": "0.82",
            "comp_min": "",
            "comp_max": "",
            "disqualified": "",
            "disqualifier_reason": "",
        }
        row.update(overrides)
        return row

    @classmethod
    def _write_csv(cls, csv_path: Path, rows: list[dict[str, str]]) -> None:
        """Write a list of row dicts to a CSV file."""
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cls._CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @pytest.fixture()
    def review(self, tmp_path: Path):
        """Temp output dir, settings, mock recorder, and active patches."""
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "jds").mkdir()

        settings = Settings(
            enabled_boards=["testboard"],
            overnight_boards=[],
            boards={
                "testboard": BoardConfig(
                    name="testboard",
                    searches=["https://example.org/search"],
                    max_pages=2,
                    headless=True,
                ),
            },
            scoring=ScoringConfig(),
            ollama=OllamaConfig(),
            output=OutputConfig(output_dir=str(out_dir)),
            chroma=ChromaConfig(persist_dir=str(tmp_path / "chroma")),
            resume_path="data/resume.md",
            archetypes_path="config/role_archetypes.toml",
        )

        recorder = MagicMock()
        recorder.get_decision = MagicMock(return_value=None)
        recorder.record = AsyncMock()

        with (
            patch("jobsearch_rag.config.load_settings", return_value=settings),
            patch("jobsearch_rag.rag.embedder.Embedder"),
            patch("jobsearch_rag.rag.store.VectorStore"),
            patch(
                "jobsearch_rag.rag.decisions.DecisionRecorder",
                return_value=recorder,
            ),
        ):
            yield {
                "out_dir": out_dir,
                "csv_path": out_dir / "results.csv",
                "recorder": recorder,
                "args": argparse.Namespace(),
            }

    # -- Tests ---------------------------------------------------------------

    def test_missing_csv_prints_message_and_exits(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When results.csv does not exist, print 'No results found' and return."""
        handle_review(review["args"])
        assert "No results found" in capsys.readouterr().out

    def test_csv_rows_are_reconstructed_as_ranked_listings(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CSV fields flow faithfully into RankedListing → display output."""
        self._write_csv(review["csv_path"], [
            self._csv_row(
                title="Data Engineer",
                company="BigTech",
                comp_min="180000",
                comp_max="250000",
                final_score="0.92",
                disqualified="true",
                disqualifier_reason="Requires clearance",
            ),
        ])
        with patch("builtins.input", side_effect=["q"]):
            handle_review(review["args"])
        out = capsys.readouterr().out
        assert "Data Engineer" in out
        assert "BigTech" in out
        assert "180,000" in out
        assert "250,000" in out
        assert "0.92" in out
        assert "DISQUALIFIED" in out
        assert "Requires clearance" in out

    def test_all_decided_prints_nothing_to_review(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When every listing has a decision, print 'nothing to review'."""
        self._write_csv(review["csv_path"], [
            self._csv_row(url="https://example.org/done-1"),
        ])
        review["recorder"].get_decision.return_value = {"verdict": "yes"}
        handle_review(review["args"])
        assert "nothing to review" in capsys.readouterr().out

    def test_undecided_count_shown_before_first_listing(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The undecided count is printed before any listing display."""
        self._write_csv(review["csv_path"], [
            self._csv_row(title="Job A", url="https://example.org/a"),
            self._csv_row(title="Job B", url="https://example.org/b"),
        ])
        with patch("builtins.input", side_effect=["q"]):
            handle_review(review["args"])
        assert "2 undecided listing(s)" in capsys.readouterr().out

    def test_quit_input_stops_review_and_prints_count(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Entering 'q' immediately stops review and reports 0 reviewed."""
        self._write_csv(review["csv_path"], [self._csv_row()])
        with patch("builtins.input", side_effect=["q"]):
            handle_review(review["args"])
        out = capsys.readouterr().out
        assert "Review stopped" in out
        assert "0 listing(s) reviewed" in out

    def test_skip_input_advances_to_next_listing(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Entering 's' advances to the next listing without recording."""
        self._write_csv(review["csv_path"], [
            self._csv_row(title="First Job", final_score="0.90",
                          url="https://example.org/j1"),
            self._csv_row(title="Second Job", final_score="0.80",
                          url="https://example.org/j2"),
        ])
        with patch("builtins.input", side_effect=["s", "q"]):
            handle_review(review["args"])
        out = capsys.readouterr().out
        assert "Second Job" in out  # skip advanced past first
        review["recorder"].record.assert_not_called()

    def test_yes_verdict_records_and_prints_confirmation(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Entering 'y' + a reason records the verdict and prints confirmation."""
        self._write_csv(review["csv_path"], [self._csv_row()])
        with patch("builtins.input", side_effect=["y", "Good fit"]):
            handle_review(review["args"])
        out = capsys.readouterr().out
        assert "Recorded: y" in out
        assert "Good fit" in out
        review["recorder"].record.assert_called_once()

    def test_verdict_without_reason_prints_short_confirmation(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Entering a verdict then Enter (empty reason) omits the reason suffix."""
        self._write_csv(review["csv_path"], [self._csv_row()])
        with patch("builtins.input", side_effect=["n", ""]):
            handle_review(review["args"])
        out = capsys.readouterr().out
        assert "Recorded: n" in out
        assert "Recorded: n —" not in out

    def test_invalid_input_reprints_help(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unrecognised input reprints the valid command list."""
        self._write_csv(review["csv_path"], [self._csv_row()])
        with patch("builtins.input", side_effect=["x", "q"]):
            handle_review(review["args"])
        assert "Invalid input" in capsys.readouterr().out

    def test_all_reviewed_prints_completion_message(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """After every undecided listing gets a verdict, print 'Review complete'."""
        self._write_csv(review["csv_path"], [self._csv_row()])
        with patch("builtins.input", side_effect=["y", ""]):
            handle_review(review["args"])
        assert "Review complete" in capsys.readouterr().out

    def test_eof_during_input_treated_as_quit(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """EOFError on input() is treated identically to entering 'q'."""
        self._write_csv(review["csv_path"], [self._csv_row()])
        with patch("builtins.input", side_effect=EOFError):
            handle_review(review["args"])
        assert "Review stopped" in capsys.readouterr().out

    def test_open_delegates_to_session_open_listing(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Entering 'o' delegates to ReviewSession.open_listing."""
        self._write_csv(review["csv_path"], [
            self._csv_row(url="https://example.org/open-me"),
        ])
        with (
            patch("builtins.input", side_effect=["o", "q"]),
            patch("jobsearch_rag.pipeline.review.webbrowser.open") as mock_wb,
        ):
            handle_review(review["args"])
        mock_wb.assert_called_once()

    def test_eof_during_reason_prompt_records_empty_reason(
        self, review: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """EOFError on the reason prompt records the verdict with an empty reason."""
        self._write_csv(review["csv_path"], [self._csv_row()])
        # First input() → "y" (verdict), second input() → EOFError (reason)
        with patch("builtins.input", side_effect=["y", EOFError]):
            handle_review(review["args"])
        out = capsys.readouterr().out
        assert "Recorded: y" in out
        assert "Recorded: y —" not in out
        review["recorder"].record.assert_called_once()

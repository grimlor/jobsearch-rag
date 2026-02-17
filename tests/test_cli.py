"""CLI handler tests — parser construction, command wiring, output formatting.

Maps to BDD specs: TestParserConstruction, TestBoardsCommand, TestIndexCommand,
TestSearchCommand, TestDecideCommand, TestExportCommand
"""

from __future__ import annotations

import argparse
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.cli import (
    build_parser,
    handle_boards,
    handle_decide,
    handle_export,
    handle_index,
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
        output=OutputConfig(),
        chroma=ChromaConfig(persist_dir=tmpdir),
        resume_path="data/resume.md",
        archetypes_path="config/role_archetypes.toml",
    )


def _make_listing(
    board: str = "testboard",
    external_id: str = "1",
    title: str = "Staff Architect",
    company: str = "Acme Corp",
) -> MagicMock:
    """Create a mock JobListing."""
    listing = MagicMock()
    listing.board = board
    listing.external_id = external_id
    listing.title = title
    listing.company = company
    listing.location = "Remote"
    listing.url = f"https://example.org/{external_id}"
    listing.full_text = "A test job description."
    return listing


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
            args = argparse.Namespace(board=None, overnight=False, open_top=None)
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
            args = argparse.Namespace(board=None, overnight=False, open_top=None)
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
            args = argparse.Namespace(board=None, overnight=False, open_top=None)
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
            args = argparse.Namespace(board="indeed", overnight=False, open_top=None)
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
            args = argparse.Namespace(board=None, overnight=False, open_top=1)
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
                    output=OutputConfig(open_top_n=0),
                    chroma=ChromaConfig(persist_dir=tmpdir),
                ),
            ),
            patch("jobsearch_rag.pipeline.runner.PipelineRunner", return_value=mock_runner),
            patch("webbrowser.open") as mock_open,
        ):
            args = argparse.Namespace(board=None, overnight=False, open_top=None)
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
            args = argparse.Namespace(job_id="nonexistent", verdict="yes")
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
            args = argparse.Namespace(job_id="zr-123", verdict="yes")
            handle_decide(args)

        mock_recorder.record.assert_awaited_once()
        output = capsys.readouterr().out
        assert "Recorded 'yes' for zr-123" in output
        assert "5" in output

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
            args = argparse.Namespace(job_id="zr-123", verdict="yes")
            handle_decide(args)

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "Could not retrieve JD text" in output


# ---------------------------------------------------------------------------
# TestExportCommand
# ---------------------------------------------------------------------------


class TestExportCommand:
    """REQUIREMENT: The export command reports its stub status clearly.

    WHO: The operator attempting to export results before Phase 5 is implemented
    WHAT: The export command prints the requested format and a clear
          message that export is not yet implemented
    WHY: Silently producing no output would leave the operator wondering
         if the command failed or if there were no results
    """

    def test_export_prints_format_and_stub_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Export prints the requested format and a 'not yet implemented' message."""
        args = argparse.Namespace(format="csv")
        handle_export(args)
        output = capsys.readouterr().out
        assert "csv" in output
        assert "not yet implemented" in output.lower()

    def test_export_accepts_all_format_choices(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each valid format is accepted without error."""
        for fmt in ("markdown", "csv", "json"):
            args = argparse.Namespace(format=fmt)
            handle_export(args)
            output = capsys.readouterr().out
            assert fmt in output

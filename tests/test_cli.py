"""CLI handler tests — parser construction, command wiring, output formatting.

Maps to BDD specs: TestParserConstruction, TestBoardsCommand, TestIndexCommand,
TestSearchCommand, TestDecideCommand, TestExportCommand
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.cli import (
    build_parser,
    handle_boards,
    handle_decide,
    handle_export,
    handle_index,
    handle_login,
    handle_reset,
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
                    output=OutputConfig(open_top_n=0),
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

"""Export tests — Markdown, CSV, and browser tab opener.

Maps to BDD specs: TestMarkdownExport, TestCSVExport, TestBrowserTabOpener

The export layer transforms ranked pipeline output into human-consumable
formats: Markdown tables (primary), CSV (spreadsheet import), and browser
tabs (immediate review).  All exporters receive ``RankedListing`` objects
produced by the Ranker — no adapter or RAG dependencies exist here.
"""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.export.browser_tabs import BrowserTabOpener
from jobsearch_rag.export.csv_export import CSVExporter
from jobsearch_rag.export.markdown import MarkdownExporter
from jobsearch_rag.pipeline.ranker import RankedListing, RankSummary
from jobsearch_rag.rag.scorer import ScoreResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _listing(
    *,
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    board: str = "ziprecruiter",
    url: str = "https://example.org/job/1",
    full_text: str = "Build distributed systems.",
    external_id: str = "ext-001",
) -> JobListing:
    return JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company=company,
        location="Remote",
        url=url,
        full_text=full_text,
    )


def _scores(
    *,
    fit: float = 0.74,
    archetype: float = 0.81,
    history: float = 0.62,
    disqualified: bool = False,
    reason: str | None = None,
) -> ScoreResult:
    return ScoreResult(
        fit_score=fit,
        archetype_score=archetype,
        history_score=history,
        disqualified=disqualified,
        disqualifier_reason=reason,
    )


def _ranked(
    *,
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    board: str = "ziprecruiter",
    url: str = "https://example.org/job/1",
    external_id: str = "ext-001",
    final_score: float = 0.78,
    fit: float = 0.74,
    archetype: float = 0.81,
    history: float = 0.62,
    disqualified: bool = False,
    reason: str | None = None,
    duplicate_boards: list[str] | None = None,
) -> RankedListing:
    return RankedListing(
        listing=_listing(
            title=title,
            company=company,
            board=board,
            url=url,
            external_id=external_id,
        ),
        scores=_scores(
            fit=fit,
            archetype=archetype,
            history=history,
            disqualified=disqualified,
            reason=reason,
        ),
        final_score=final_score,
        duplicate_boards=duplicate_boards or [],
    )


def _summary(
    *,
    total_found: int = 10,
    total_scored: int = 8,
    total_excluded: int = 1,
    total_deduplicated: int = 1,
) -> RankSummary:
    return RankSummary(
        total_found=total_found,
        total_scored=total_scored,
        total_excluded=total_excluded,
        total_deduplicated=total_deduplicated,
    )


# ---------------------------------------------------------------------------
# TestMarkdownExport
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    """REQUIREMENT: Markdown output is human-readable and complete.

    WHO: The operator reviewing results in Obsidian or a text editor
    WHAT: Output includes title, company, board, final score, component scores,
          disqualifier status, and a clickable URL; disqualified roles are
          omitted from output (score 0.0 = excluded); results are sorted
          descending by final score; run summary appears at top
    WHY: The ranked list is the primary product of every run —
         missing fields or wrong sort order defeats the purpose
    """

    def test_output_includes_all_required_fields_per_listing(self, tmp_path: Path) -> None:
        """Each listing in Markdown output includes title, company, board, scores, disqualifier status, and URL."""
        listings = [_ranked()]
        out = tmp_path / "results.md"
        MarkdownExporter().export(listings, str(out), summary=_summary())
        content = out.read_text()
        assert "Staff Architect" in content
        assert "Acme Corp" in content
        assert "ziprecruiter" in content
        assert "0.78" in content  # final score
        assert "example.org/job/1" in content
        assert "Not disqualified" in content

    def test_listings_are_sorted_descending_by_final_score(self, tmp_path: Path) -> None:
        """Listings appear in descending final_score order so the best matches are reviewed first."""
        listings = [
            _ranked(title="Low", final_score=0.40),
            _ranked(title="High", final_score=0.90, external_id="ext-002", url="https://example.org/job/2"),
            _ranked(title="Mid", final_score=0.65, external_id="ext-003", url="https://example.org/job/3"),
        ]
        out = tmp_path / "results.md"
        MarkdownExporter().export(listings, str(out), summary=_summary())
        content = out.read_text()
        high_pos = content.index("High")
        mid_pos = content.index("Mid")
        low_pos = content.index("Low")
        assert high_pos < mid_pos < low_pos

    def test_disqualified_roles_are_not_present_in_output(self, tmp_path: Path) -> None:
        """Disqualified roles (score 0.0) are excluded entirely from the Markdown output."""
        listings = [
            _ranked(title="Good Role", final_score=0.80),
            _ranked(
                title="Bad Role",
                final_score=0.0,
                disqualified=True,
                reason="IC role",
                external_id="ext-002",
                url="https://example.org/job/2",
            ),
        ]
        out = tmp_path / "results.md"
        MarkdownExporter().export(listings, str(out), summary=_summary())
        content = out.read_text()
        assert "Good Role" in content
        assert "Bad Role" not in content

    def test_score_explanation_shows_all_three_component_scores(self, tmp_path: Path) -> None:
        """The score explanation shows fit, archetype, and history component values for transparency."""
        listings = [_ranked(fit=0.74, archetype=0.81, history=0.62)]
        out = tmp_path / "results.md"
        MarkdownExporter().export(listings, str(out), summary=_summary())
        content = out.read_text()
        assert "0.74" in content  # fit
        assert "0.81" in content  # archetype
        assert "0.62" in content  # history

    def test_run_summary_appears_at_top_of_output(self, tmp_path: Path) -> None:
        """The run summary header appears before any listing, providing immediate run context."""
        listings = [_ranked()]
        out = tmp_path / "results.md"
        MarkdownExporter().export(listings, str(out), summary=_summary())
        content = out.read_text()
        # Summary section should appear before the first listing title
        summary_pos = content.index("Summary")
        listing_pos = content.index("Staff Architect")
        assert summary_pos < listing_pos

    def test_run_summary_includes_total_found_scored_excluded_deduplicated(
        self, tmp_path: Path
    ) -> None:
        """The summary reports total found, scored, excluded, and deduplicated counts for full audit trail."""
        listings = [_ranked()]
        summary = _summary(total_found=10, total_scored=8, total_excluded=1, total_deduplicated=1)
        out = tmp_path / "results.md"
        MarkdownExporter().export(listings, str(out), summary=summary)
        content = out.read_text()
        assert "10" in content  # total found
        assert "8" in content   # total scored
        # excluded and deduplicated counts
        assert "1" in content

    def test_url_is_present_and_non_empty_for_every_listing(self, tmp_path: Path) -> None:
        """Every listing has a non-empty URL so the operator can click through to the source posting."""
        listings = [
            _ranked(url="https://example.org/job/1"),
            _ranked(url="https://example.org/job/2", external_id="ext-002"),
        ]
        out = tmp_path / "results.md"
        MarkdownExporter().export(listings, str(out), summary=_summary())
        content = out.read_text()
        assert "https://example.org/job/1" in content
        assert "https://example.org/job/2" in content

    def test_empty_result_set_produces_output_with_summary_and_no_table(
        self, tmp_path: Path
    ) -> None:
        """An empty result set produces a valid Markdown file with a summary but no listing table."""
        out = tmp_path / "results.md"
        summary = _summary(total_found=0, total_scored=0, total_excluded=0, total_deduplicated=0)
        MarkdownExporter().export([], str(out), summary=summary)
        content = out.read_text()
        assert "Summary" in content
        assert "No results" in content or "0" in content


# ---------------------------------------------------------------------------
# TestCSVExport
# ---------------------------------------------------------------------------


class TestCSVExport:
    """REQUIREMENT: CSV export is valid and importable by standard tools.

    WHO: The operator importing results into a spreadsheet or ATS tracker
    WHAT: Output is valid CSV with a header row; all required columns are present;
          JD text is excluded from CSV (too large); special characters in
          company names or titles do not break CSV formatting
    WHY: A CSV with unescaped commas or missing headers silently corrupts
         on import — the operator may not notice
    """

    def test_csv_has_header_row_with_all_required_columns(self, tmp_path: Path) -> None:
        """The CSV starts with a header row containing all required column names for import tools."""
        listings = [_ranked()]
        out = tmp_path / "results.csv"
        CSVExporter().export(listings, str(out), summary=_summary())
        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
        required = ["title", "company", "board", "final_score", "url"]
        for col in required:
            assert col in header, f"Missing required column: {col}"

    def test_full_jd_text_is_not_included_in_csv_output(self, tmp_path: Path) -> None:
        """full_text is omitted from CSV since it is too large for spreadsheet cells."""
        listings = [_ranked()]
        out = tmp_path / "results.csv"
        CSVExporter().export(listings, str(out), summary=_summary())
        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "full_text" not in header

    def test_company_names_with_commas_are_properly_quoted(self, tmp_path: Path) -> None:
        """Company names containing commas are CSV-quoted so they don't split across columns on import."""
        listings = [_ranked(company="Acme, Inc.")]
        out = tmp_path / "results.csv"
        CSVExporter().export(listings, str(out), summary=_summary())
        with open(out) as f:
            reader = csv.reader(f)
            _header = next(reader)
            row = next(reader)
        # The company name should survive a round-trip through csv.reader intact
        assert "Acme, Inc." in row

    def test_empty_result_set_produces_header_only_csv(self, tmp_path: Path) -> None:
        """An empty result set produces a CSV with headers only, remaining valid for import tools."""
        out = tmp_path / "results.csv"
        summary = _summary(total_found=0, total_scored=0, total_excluded=0, total_deduplicated=0)
        CSVExporter().export([], str(out), summary=summary)
        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert len(header) > 0
        assert len(rows) == 0

    def test_csv_row_count_matches_scored_listing_count(self, tmp_path: Path) -> None:
        """The number of CSV data rows equals the number of scored (non-disqualified) listings."""
        listings = [
            _ranked(title="Role A", external_id="ext-001"),
            _ranked(title="Role B", external_id="ext-002", url="https://example.org/job/2"),
            _ranked(
                title="Disqualified",
                final_score=0.0,
                disqualified=True,
                reason="IC role",
                external_id="ext-003",
                url="https://example.org/job/3",
            ),
        ]
        out = tmp_path / "results.csv"
        CSVExporter().export(listings, str(out), summary=_summary())
        with open(out) as f:
            reader = csv.reader(f)
            _header = next(reader)
            rows = list(reader)
        # 2 scored, 1 disqualified excluded
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# TestBrowserTabOpener
# ---------------------------------------------------------------------------


class TestBrowserTabOpener:
    """REQUIREMENT: Top-ranked results open as browser tabs in score order.

    WHO: The operator who wants to review shortlisted roles without manually
         clicking through the ranked output
    WHAT: Tabs open in descending score order; the count respects --open-top N
          or the settings.toml default; fewer available results than N opens
          only what exists without error; disqualified roles are never opened;
          a failed tab open logs the URL and continues to the next
    WHY: The tab opener is the last mile of the workflow — an error here
         that aborts remaining tabs wastes the entire run's value;
         opening in wrong order defeats the purpose of ranking
    """

    @patch("webbrowser.open")
    def test_tabs_open_in_descending_score_order(self, mock_open: object) -> None:
        """Browser tabs open highest-scored listing first so the best match is immediately visible."""
        import webbrowser as _wb

        listings = [
            _ranked(title="Low", final_score=0.40, url="https://example.org/low"),
            _ranked(title="High", final_score=0.90, url="https://example.org/high", external_id="ext-002"),
            _ranked(title="Mid", final_score=0.65, url="https://example.org/mid", external_id="ext-003"),
        ]
        BrowserTabOpener().open(listings, top_n=3)
        calls = [c.args[0] for c in _wb.open.call_args_list]  # type: ignore[union-attr]
        assert calls == [
            "https://example.org/high",
            "https://example.org/mid",
            "https://example.org/low",
        ]

    @patch("webbrowser.open")
    def test_tab_count_respects_open_top_n_from_cli(self, mock_open: object) -> None:
        """The --open-top N CLI flag limits the number of tabs opened to exactly N."""
        import webbrowser as _wb

        listings = [
            _ranked(
                title=f"Role {i}",
                final_score=0.9 - i * 0.1,
                external_id=f"ext-{i}",
                url=f"https://example.org/{i}",
            )
            for i in range(5)
        ]
        BrowserTabOpener().open(listings, top_n=2)
        assert _wb.open.call_count == 2  # type: ignore[union-attr]

    @patch("webbrowser.open")
    def test_tab_count_respects_open_top_n_from_settings_when_cli_not_provided(
        self, mock_open: object
    ) -> None:
        """When --open-top is not specified on CLI, the default from settings.toml is used."""
        import webbrowser as _wb

        listings = [
            _ranked(
                title=f"Role {i}",
                final_score=0.9 - i * 0.1,
                external_id=f"ext-{i}",
                url=f"https://example.org/{i}",
            )
            for i in range(10)
        ]
        # Default top_n=5
        BrowserTabOpener().open(listings)
        assert _wb.open.call_count == 5  # type: ignore[union-attr]

    @patch("webbrowser.open")
    def test_fewer_results_than_n_opens_all_available_without_error(
        self, mock_open: object
    ) -> None:
        """If fewer results exist than N, all are opened without raising on the shortfall."""
        import webbrowser as _wb

        listings = [
            _ranked(title="Only One", url="https://example.org/only"),
        ]
        BrowserTabOpener().open(listings, top_n=10)
        assert _wb.open.call_count == 1  # type: ignore[union-attr]

    @patch("webbrowser.open")
    def test_disqualified_roles_are_never_opened_as_tabs(self, mock_open: object) -> None:
        """Disqualified roles are excluded from tab opening regardless of their position in the list."""
        import webbrowser as _wb

        listings = [
            _ranked(title="Good", final_score=0.80, url="https://example.org/good"),
            _ranked(
                title="Bad",
                final_score=0.0,
                disqualified=True,
                reason="IC role",
                url="https://example.org/bad",
                external_id="ext-002",
            ),
        ]
        BrowserTabOpener().open(listings, top_n=5)
        calls = [c.args[0] for c in _wb.open.call_args_list]  # type: ignore[union-attr]
        assert "https://example.org/good" in calls
        assert "https://example.org/bad" not in calls

    @patch("webbrowser.open", side_effect=[OSError("browser failed"), None])
    def test_failed_tab_open_logs_url_and_continues_to_next(
        self, mock_open: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed tab open logs the URL and proceeds to the next — one failure doesn't abort the rest."""
        listings = [
            _ranked(title="Fail", final_score=0.90, url="https://example.org/fail"),
            _ranked(title="OK", final_score=0.80, url="https://example.org/ok", external_id="ext-002"),
        ]
        with caplog.at_level(logging.WARNING):
            BrowserTabOpener().open(listings, top_n=2)
        assert "example.org/fail" in caplog.text

    @patch("webbrowser.open")
    def test_zero_scored_results_opens_no_tabs_and_logs_advisory(
        self, mock_open: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When all results score zero, no tabs open and an advisory message is logged."""
        import webbrowser as _wb

        listings = [
            _ranked(title="Zero", final_score=0.0, disqualified=True, reason="IC role"),
        ]
        with caplog.at_level(logging.INFO):
            BrowserTabOpener().open(listings, top_n=5)
        assert _wb.open.call_count == 0  # type: ignore[union-attr]
        assert "no" in caplog.text.lower() or "0" in caplog.text

    @patch("webbrowser.open")
    def test_tabs_use_default_system_browser_not_playwright_session(
        self, mock_open: object
    ) -> None:
        """Tabs open in the system browser, not in the Playwright automation session."""
        import webbrowser as _wb

        listings = [_ranked()]
        BrowserTabOpener().open(listings, top_n=1)
        # Verify webbrowser.open was called (not Playwright)
        _wb.open.assert_called_once()  # type: ignore[union-attr]

    @patch("webbrowser.open")
    def test_open_top_zero_opens_no_tabs_without_error(self, mock_open: object) -> None:
        """--open-top 0 is a valid choice that skips tab opening entirely without raising."""
        import webbrowser as _wb

        listings = [_ranked()]
        BrowserTabOpener().open(listings, top_n=0)
        assert _wb.open.call_count == 0  # type: ignore[union-attr]

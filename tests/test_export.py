"""BDD specs for the export layer — Markdown, CSV, and browser tab output.

Covers:
    - TestMarkdownExport  (BDD Specifications — export.md)
    - TestCSVExport       (BDD Specifications — export.md)
    - TestBrowserTabOpener (BDD Specifications — export.md)

Each test class verifies one export format. All classes share the same
approach: construct real RankedListing instances with explicit ScoreResult
values, pass them through the real exporter, and assert on the written
output. No exporters are mocked.
"""

# Public API surface (from src/jobsearch_rag/export/markdown.py):
#   MarkdownExporter()
#   exporter.export(listings, output_path, *, summary=None) -> None
#
# Public API surface (from src/jobsearch_rag/export/csv_export.py):
#   CSVExporter()
#   exporter.export(listings, output_path, *, summary=None) -> None
#
# Public API surface (from src/jobsearch_rag/export/browser_tabs.py):
#   BrowserTabOpener()
#   opener.open(listings, top_n=5) -> None
#
# Public API surface (from src/jobsearch_rag/pipeline/ranker.py):
#   RankedListing(listing, scores, final_score, duplicate_boards=[])
#   RankedListing.score_explanation() -> str
#   RankSummary(total_found, total_scored, total_excluded, total_deduplicated)
#
# Public API surface (from src/jobsearch_rag/rag/scorer.py):
#   ScoreResult(fit_score, archetype_score, history_score, disqualified,
#               disqualifier_reason=None, comp_score=0.5, negative_score=0.0,
#               culture_score=0.0)
#
# Public API surface (from src/jobsearch_rag/adapters/base.py):
#   JobListing(board, external_id, title, company, location, url, full_text,
#              posted_at=None, raw_html=None, comp_min=None, comp_max=None,
#              comp_source=None, comp_text=None, metadata={})

from __future__ import annotations

import csv
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(
    *,
    board: str = "ziprecruiter",
    external_id: str = "1",
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    location: str = "Remote",
    url: str = "https://ziprecruiter.com/1",
    full_text: str = "A detailed job description for a staff architect role.",
    comp_min: float | None = None,
    comp_max: float | None = None,
) -> JobListing:
    return JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company=company,
        location=location,
        url=url,
        full_text=full_text,
        comp_min=comp_min,
        comp_max=comp_max,
    )


def _make_scores(
    *,
    fit_score: float = 0.8,
    archetype_score: float = 0.7,
    history_score: float = 0.5,
    comp_score: float = 0.6,
    culture_score: float = 0.3,
    negative_score: float = 0.1,
    disqualified: bool = False,
    disqualifier_reason: str | None = None,
) -> ScoreResult:
    return ScoreResult(
        fit_score=fit_score,
        archetype_score=archetype_score,
        history_score=history_score,
        comp_score=comp_score,
        culture_score=culture_score,
        negative_score=negative_score,
        disqualified=disqualified,
        disqualifier_reason=disqualifier_reason,
    )


def _make_ranked(
    *,
    listing: JobListing | None = None,
    scores: ScoreResult | None = None,
    final_score: float = 0.85,
) -> RankedListing:
    return RankedListing(
        listing=listing or _make_listing(),
        scores=scores or _make_scores(),
        final_score=final_score,
    )


# ---------------------------------------------------------------------------
# TestMarkdownExport
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    """
    REQUIREMENT: Markdown output is human-readable, complete, and correctly ordered.

    WHO: The operator reviewing results in Obsidian or a text editor
    WHAT: Output includes title, company, board, final score, all six
          component scores (archetype, fit, culture, history, comp, negative),
          disqualifier status, and a clickable URL per listing; results are
          sorted descending by final score; run summary appears at top;
          empty result set produces output with summary and no table
    WHY: The ranked list is the primary product of every run —
         missing fields or wrong sort order defeats the purpose of the tool

    MOCK BOUNDARY:
        Mock:  nothing — MarkdownExporter takes RankedListing instances and
               writes to tmp_path; no I/O boundaries to mock
        Real:  MarkdownExporter instance, RankedListing instances constructed
               with real ScoreResult values, output file written to tmp_path
        Never: Mock the exporter; construct RankedListing and ScoreResult
               with explicit field values (not MagicMock) so output content
               is deterministic and verifiable
    """

    def test_output_includes_all_required_fields_per_listing(
        self, tmp_path: Path
    ) -> None:
        """
        When a single ranked listing is exported to Markdown
        Then the output contains the listing's title, company, board,
             final score, score breakdown, and URL
        """
        # Given: a single ranked listing with known values
        listing = _make_listing(
            title="Staff Architect",
            company="Acme Corp",
            board="ziprecruiter",
            url="https://ziprecruiter.com/1",
        )
        ranked = _make_ranked(listing=listing, final_score=0.85)
        output_path = str(tmp_path / "results.md")

        # When: the exporter writes the file
        exporter = MarkdownExporter()
        exporter.export([ranked], output_path)

        # Then: all required fields appear in the output
        content = (tmp_path / "results.md").read_text()
        assert "Staff Architect" in content, (
            f"Title 'Staff Architect' not found in output. Got:\n{content}"
        )
        assert "Acme Corp" in content, (
            f"Company 'Acme Corp' not found in output. Got:\n{content}"
        )
        assert "ziprecruiter" in content, (
            f"Board 'ziprecruiter' not found in output. Got:\n{content}"
        )
        assert "0.85" in content, (
            f"Final score '0.85' not found in output. Got:\n{content}"
        )
        assert "https://ziprecruiter.com/1" in content, (
            f"URL not found in output. Got:\n{content}"
        )
        # Score breakdown should appear (from score_explanation)
        assert "Fit:" in content, (
            f"Score breakdown 'Fit:' not found in output. Got:\n{content}"
        )

    def test_listings_are_sorted_descending_by_final_score(
        self, tmp_path: Path
    ) -> None:
        """
        Given three listings with different final scores
        When they are exported to Markdown
        Then they appear in descending score order
        """
        # Given: three listings with known, distinct final scores
        high = _make_ranked(
            listing=_make_listing(title="High Scorer", external_id="h"),
            final_score=0.95,
        )
        mid = _make_ranked(
            listing=_make_listing(title="Mid Scorer", external_id="m"),
            final_score=0.70,
        )
        low = _make_ranked(
            listing=_make_listing(title="Low Scorer", external_id="l"),
            final_score=0.50,
        )
        output_path = str(tmp_path / "results.md")

        # When: exported in non-sorted order
        exporter = MarkdownExporter()
        exporter.export([low, high, mid], output_path)

        # Then: they appear high → mid → low in the file
        content = (tmp_path / "results.md").read_text()
        pos_high = content.index("High Scorer")
        pos_mid = content.index("Mid Scorer")
        pos_low = content.index("Low Scorer")
        assert pos_high < pos_mid < pos_low, (
            f"Expected descending score order. Positions: "
            f"High={pos_high}, Mid={pos_mid}, Low={pos_low}"
        )

    def test_disqualified_roles_are_not_present_in_output(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing that is disqualified (disqualified=True, final_score=0.0)
        When exported to Markdown
        Then the disqualified listing does not appear in the output
        """
        # Given: one qualified and one disqualified listing
        qualified = _make_ranked(
            listing=_make_listing(title="Good Role", external_id="g"),
            final_score=0.80,
        )
        disqualified = _make_ranked(
            listing=_make_listing(title="Bad Role", external_id="b"),
            scores=_make_scores(disqualified=True, disqualifier_reason="Staffing agency"),
            final_score=0.0,
        )
        output_path = str(tmp_path / "results.md")

        # When: both are exported
        exporter = MarkdownExporter()
        exporter.export([qualified, disqualified], output_path)

        # Then: only the qualified listing appears
        content = (tmp_path / "results.md").read_text()
        assert "Good Role" in content, (
            f"Qualified listing 'Good Role' should be in output. Got:\n{content}"
        )
        assert "Bad Role" not in content, (
            f"Disqualified listing 'Bad Role' should NOT be in output. Got:\n{content}"
        )

    def test_score_breakdown_shows_all_six_component_scores(
        self, tmp_path: Path
    ) -> None:
        """
        When a listing is exported to Markdown
        Then the score breakdown includes Archetype, Fit, History, Comp,
             Culture, and Negative labels
        """
        # Given: a listing with all six component scores set
        ranked = _make_ranked(final_score=0.85)
        output_path = str(tmp_path / "results.md")

        # When: exported
        exporter = MarkdownExporter()
        exporter.export([ranked], output_path)

        # Then: all six component labels appear in the breakdown
        content = (tmp_path / "results.md").read_text()
        for label in ("Archetype:", "Fit:", "History:", "Comp:", "Culture:", "Negative:"):
            assert label in content, (
                f"Score component '{label}' not found in output. Got:\n{content}"
            )

    def test_score_breakdown_includes_culture_score(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing with culture_score=0.45
        When exported to Markdown
        Then the breakdown shows "Culture: 0.45"
        """
        # Given: a listing with a specific culture score
        scores = _make_scores(culture_score=0.45)
        ranked = _make_ranked(scores=scores, final_score=0.80)
        output_path = str(tmp_path / "results.md")

        # When: exported
        exporter = MarkdownExporter()
        exporter.export([ranked], output_path)

        # Then: the culture score value appears in the breakdown
        content = (tmp_path / "results.md").read_text()
        assert "Culture: 0.45" in content, (
            f"Expected 'Culture: 0.45' in output. Got:\n{content}"
        )

    def test_score_breakdown_includes_negative_score(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing with negative_score=0.22
        When exported to Markdown
        Then the breakdown shows "Negative: 0.22"
        """
        # Given: a listing with a specific negative score
        scores = _make_scores(negative_score=0.22)
        ranked = _make_ranked(scores=scores, final_score=0.75)
        output_path = str(tmp_path / "results.md")

        # When: exported
        exporter = MarkdownExporter()
        exporter.export([ranked], output_path)

        # Then: the negative score value appears in the breakdown
        content = (tmp_path / "results.md").read_text()
        assert "Negative: 0.22" in content, (
            f"Expected 'Negative: 0.22' in output. Got:\n{content}"
        )

    def test_score_breakdown_includes_comp_score_and_salary_range(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing with comp_score=0.90 and comp_min/comp_max set
        When exported to Markdown
        Then the breakdown shows "Comp: 0.90"
        """
        # Given: a listing with comp data
        listing = _make_listing(comp_min=150000.0, comp_max=200000.0)
        scores = _make_scores(comp_score=0.90)
        ranked = _make_ranked(listing=listing, scores=scores, final_score=0.88)
        output_path = str(tmp_path / "results.md")

        # When: exported
        exporter = MarkdownExporter()
        exporter.export([ranked], output_path)

        # Then: the comp score appears in the breakdown
        content = (tmp_path / "results.md").read_text()
        assert "Comp: 0.90" in content, (
            f"Expected 'Comp: 0.90' in output. Got:\n{content}"
        )

    def test_run_summary_appears_at_top(self, tmp_path: Path) -> None:
        """
        When a listing is exported with a RankSummary
        Then the run summary section appears before any listing data
        """
        # Given: a listing with a summary
        ranked = _make_ranked(final_score=0.80)
        summary = RankSummary(
            total_found=10, total_scored=8, total_excluded=1, total_deduplicated=1
        )
        output_path = str(tmp_path / "results.md")

        # When: exported with summary
        exporter = MarkdownExporter()
        exporter.export([ranked], output_path, summary=summary)

        # Then: the summary heading appears before the listing title
        content = (tmp_path / "results.md").read_text()
        summary_pos = content.index("# Run Summary")
        listing_pos = content.index("Staff Architect")
        assert summary_pos < listing_pos, (
            f"Run summary (pos={summary_pos}) should appear before "
            f"listing data (pos={listing_pos})"
        )

    def test_run_summary_includes_found_scored_excluded_deduped_counts(
        self, tmp_path: Path
    ) -> None:
        """
        Given a RankSummary with known counts
        When exported to Markdown
        Then the summary includes Total found, Total scored, Excluded, and
             Deduplicated counts
        """
        # Given: a summary with specific counts
        ranked = _make_ranked(final_score=0.80)
        summary = RankSummary(
            total_found=25, total_scored=20, total_excluded=3, total_deduplicated=2
        )
        output_path = str(tmp_path / "results.md")

        # When: exported
        exporter = MarkdownExporter()
        exporter.export([ranked], output_path, summary=summary)

        # Then: all four count fields appear in the summary
        content = (tmp_path / "results.md").read_text()
        assert "25" in content, (
            f"Total found count '25' not found in summary. Got:\n{content}"
        )
        assert "20" in content, (
            f"Total scored count '20' not found in summary. Got:\n{content}"
        )
        assert "3" in content, (
            f"Excluded count '3' not found in summary. Got:\n{content}"
        )
        assert "2" in content, (
            f"Deduplicated count '2' not found in summary. Got:\n{content}"
        )

    def test_url_is_present_and_non_empty_for_every_listing(
        self, tmp_path: Path
    ) -> None:
        """
        Given two listings with different URLs
        When exported to Markdown
        Then both URLs appear in the output
        """
        # Given: two listings with distinct URLs
        r1 = _make_ranked(
            listing=_make_listing(
                title="Role A", external_id="a", url="https://board.com/job-a"
            ),
            final_score=0.90,
        )
        r2 = _make_ranked(
            listing=_make_listing(
                title="Role B", external_id="b", url="https://board.com/job-b"
            ),
            final_score=0.80,
        )
        output_path = str(tmp_path / "results.md")

        # When: exported
        exporter = MarkdownExporter()
        exporter.export([r1, r2], output_path)

        # Then: both URLs are present
        content = (tmp_path / "results.md").read_text()
        assert "https://board.com/job-a" in content, (
            f"URL for Role A not found. Got:\n{content}"
        )
        assert "https://board.com/job-b" in content, (
            f"URL for Role B not found. Got:\n{content}"
        )

    def test_empty_result_set_produces_summary_without_listings(
        self, tmp_path: Path
    ) -> None:
        """
        Given an empty list of ranked listings
        When exported to Markdown with a summary
        Then the output contains the run summary but no listing table
        """
        # Given: empty listings with a summary
        summary = RankSummary(
            total_found=0, total_scored=0, total_excluded=0, total_deduplicated=0
        )
        output_path = str(tmp_path / "results.md")

        # When: exported with no listings
        exporter = MarkdownExporter()
        exporter.export([], output_path, summary=summary)

        # Then: summary is present but no table header
        content = (tmp_path / "results.md").read_text()
        assert "# Run Summary" in content, (
            f"Run summary heading should be present. Got:\n{content}"
        )
        assert "Ranked Listings" not in content, (
            f"Listing table should not appear for empty results. Got:\n{content}"
        )


# ---------------------------------------------------------------------------
# TestCSVExport
# ---------------------------------------------------------------------------


class TestCSVExport:
    """
    REQUIREMENT: CSV export is valid and importable by standard tools.

    WHO: The operator importing results into a spreadsheet or ATS tracker
    WHAT: Output is valid CSV with a header row; all required columns are
          present including comp_score, comp_min, comp_max, culture_score,
          and negative_score; JD text is excluded; special characters in
          company names or titles are properly quoted
    WHY: A CSV with unescaped commas or missing headers silently corrupts
         on import — the operator may not notice until decisions are made
         on incomplete data

    MOCK BOUNDARY:
        Mock:  nothing — CSVExporter takes RankedListing instances and writes
               to tmp_path; no I/O boundaries to mock
        Real:  CSVExporter instance, RankedListing instances with real values,
               output file written to tmp_path
        Never: Mock the exporter; construct RankedListing with explicit values
               including company names containing commas to verify quoting
    """

    def test_csv_has_header_row_with_all_required_columns(
        self, tmp_path: Path
    ) -> None:
        """
        When a single listing is exported to CSV
        Then the first row contains all required column headers
        """
        # Given: a single ranked listing
        ranked = _make_ranked(final_score=0.85)
        output_path = str(tmp_path / "results.csv")

        # When: exported to CSV
        exporter = CSVExporter()
        exporter.export([ranked], output_path)

        # Then: the header row contains all expected columns
        with open(output_path) as f:
            reader = csv.reader(f)
            headers = next(reader)

        required = [
            "title", "company", "board", "location", "final_score",
            "fit_score", "archetype_score", "history_score", "comp_score",
            "culture_score", "negative_score", "comp_min", "comp_max",
            "disqualified", "disqualifier_reason", "url",
        ]
        for col in required:
            assert col in headers, (
                f"Required column '{col}' not in CSV header. "
                f"Got: {headers}"
            )

    def test_csv_includes_comp_score_comp_min_comp_max_columns(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing with comp_min=150000 and comp_max=200000
        When exported to CSV
        Then the comp_score, comp_min, and comp_max columns contain values
        """
        # Given: a listing with compensation data
        listing = _make_listing(comp_min=150000.0, comp_max=200000.0)
        scores = _make_scores(comp_score=0.90)
        ranked = _make_ranked(listing=listing, scores=scores, final_score=0.88)
        output_path = str(tmp_path / "results.csv")

        # When: exported
        exporter = CSVExporter()
        exporter.export([ranked], output_path)

        # Then: comp columns have non-empty values
        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["comp_score"] != "", (
            f"comp_score should be populated. Got: {row['comp_score']}"
        )
        assert row["comp_min"] == "150000", (
            f"comp_min should be '150000'. Got: {row['comp_min']}"
        )
        assert row["comp_max"] == "200000", (
            f"comp_max should be '200000'. Got: {row['comp_max']}"
        )

    def test_csv_includes_culture_score_column(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing with culture_score=0.45
        When exported to CSV
        Then the culture_score column contains "0.4500"
        """
        # Given: a listing with a specific culture score
        scores = _make_scores(culture_score=0.45)
        ranked = _make_ranked(scores=scores, final_score=0.80)
        output_path = str(tmp_path / "results.csv")

        # When: exported
        exporter = CSVExporter()
        exporter.export([ranked], output_path)

        # Then: the culture_score column is populated
        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["culture_score"] == "0.4500", (
            f"culture_score should be '0.4500'. Got: {row['culture_score']}"
        )

    def test_csv_includes_negative_score_column(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing with negative_score=0.22
        When exported to CSV
        Then the negative_score column contains "0.2200"
        """
        # Given: a listing with a specific negative score
        scores = _make_scores(negative_score=0.22)
        ranked = _make_ranked(scores=scores, final_score=0.75)
        output_path = str(tmp_path / "results.csv")

        # When: exported
        exporter = CSVExporter()
        exporter.export([ranked], output_path)

        # Then: the negative_score column is populated
        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["negative_score"] == "0.2200", (
            f"negative_score should be '0.2200'. Got: {row['negative_score']}"
        )

    def test_full_jd_text_is_not_included_in_csv(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing with distinctive full_text content
        When exported to CSV
        Then the full JD text does not appear in any column
        """
        # Given: a listing with a uniquely identifiable full_text
        listing = _make_listing(
            full_text="UNIQUE_JD_MARKER: This is a long job description that "
            "should never appear in any CSV column."
        )
        ranked = _make_ranked(listing=listing, final_score=0.80)
        output_path = str(tmp_path / "results.csv")

        # When: exported
        exporter = CSVExporter()
        exporter.export([ranked], output_path)

        # Then: the JD text is absent from the file content
        content = (tmp_path / "results.csv").read_text()
        assert "UNIQUE_JD_MARKER" not in content, (
            f"Full JD text should not appear in CSV. Got:\n{content}"
        )

    def test_company_names_with_commas_are_properly_quoted(
        self, tmp_path: Path
    ) -> None:
        """
        Given a listing whose company name contains a comma
        When exported to CSV
        Then the CSV is still valid — the company field is properly quoted
        """
        # Given: a company name with a comma
        listing = _make_listing(company="Acme, Inc.")
        ranked = _make_ranked(listing=listing, final_score=0.80)
        output_path = str(tmp_path / "results.csv")

        # When: exported
        exporter = CSVExporter()
        exporter.export([ranked], output_path)

        # Then: read back via csv module — the company field is intact
        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["company"] == "Acme, Inc.", (
            f"Company name with comma should be preserved. Got: {row['company']}"
        )

    def test_empty_result_set_produces_header_only_csv(
        self, tmp_path: Path
    ) -> None:
        """
        Given an empty list of ranked listings
        When exported to CSV
        Then the file contains only the header row
        """
        # Given: no listings
        output_path = str(tmp_path / "results.csv")

        # When: exported with empty list
        exporter = CSVExporter()
        exporter.export([], output_path)

        # Then: only the header row exists
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 1, (
            f"Expected exactly 1 line (header only) for empty results. "
            f"Got {len(lines)} lines"
        )

    def test_csv_row_count_matches_scored_listing_count(
        self, tmp_path: Path
    ) -> None:
        """
        Given three qualified listings
        When exported to CSV
        Then the CSV has a header plus three data rows
        """
        # Given: three distinct listings
        listings = [
            _make_ranked(
                listing=_make_listing(title=f"Role {i}", external_id=str(i)),
                final_score=0.9 - (i * 0.1),
            )
            for i in range(3)
        ]
        output_path = str(tmp_path / "results.csv")

        # When: exported
        exporter = CSVExporter()
        exporter.export(listings, output_path)

        # Then: 1 header + 3 data rows = 4 lines
        with open(output_path) as f:
            reader = csv.reader(f)
            all_rows = list(reader)

        data_rows = all_rows[1:]  # skip header
        assert len(data_rows) == 3, (
            f"Expected 3 data rows for 3 listings. "
            f"Got {len(data_rows)} rows"
        )


# ---------------------------------------------------------------------------
# TestBrowserTabOpener
# ---------------------------------------------------------------------------


class TestBrowserTabOpener:
    """
    REQUIREMENT: Top-ranked results open as browser tabs in score order.

    WHO: The operator who wants to review shortlisted roles without manually
         clicking through the ranked output
    WHAT: Tabs open in descending score order; count respects --open-top N
          or settings default; fewer results than N opens only what exists;
          disqualified roles are never opened; a failed tab open logs the
          URL and continues to the next
    WHY: Opening in wrong order defeats the purpose of ranking;
         an error aborting remaining tabs discards the run's value

    MOCK BOUNDARY:
        Mock:  webbrowser.open (patch to capture calls without opening
               a real browser); logging infrastructure via tmp_path
        Real:  BrowserTabOpener instance, RankedListing instances with
               real URL and score values
        Never: Mock BrowserTabOpener itself; verify tab order by inspecting
               the sequence of URLs passed to the webbrowser.open mock
    """

    def test_tabs_open_in_descending_score_order(self) -> None:
        """
        Given three listings with different final scores
        When the browser tab opener is invoked with top_n=3
        Then webbrowser.open is called in descending score order
        """
        # Given: three listings with known scores, passed in non-sorted order
        low = _make_ranked(
            listing=_make_listing(
                title="Low", external_id="l", url="https://board.com/low"
            ),
            final_score=0.50,
        )
        high = _make_ranked(
            listing=_make_listing(
                title="High", external_id="h", url="https://board.com/high"
            ),
            final_score=0.95,
        )
        mid = _make_ranked(
            listing=_make_listing(
                title="Mid", external_id="m", url="https://board.com/mid"
            ),
            final_score=0.70,
        )

        # When: open is called with all three
        with patch("jobsearch_rag.export.browser_tabs.webbrowser.open") as mock_open:
            opener = BrowserTabOpener()
            opener.open([low, high, mid], top_n=3)

        # Then: calls are in descending score order
        called_urls = [call.args[0] for call in mock_open.call_args_list]
        assert called_urls == [
            "https://board.com/high",
            "https://board.com/mid",
            "https://board.com/low",
        ], (
            f"Expected tabs in descending score order. Got: {called_urls}"
        )

    def test_tab_count_respects_open_top_n_from_cli(self) -> None:
        """
        Given five listings and top_n=2
        When the browser tab opener is invoked
        Then only 2 tabs are opened
        """
        # Given: five listings
        listings = [
            _make_ranked(
                listing=_make_listing(
                    title=f"Role {i}", external_id=str(i),
                    url=f"https://board.com/{i}",
                ),
                final_score=0.9 - (i * 0.05),
            )
            for i in range(5)
        ]

        # When: open with top_n=2
        with patch("jobsearch_rag.export.browser_tabs.webbrowser.open") as mock_open:
            opener = BrowserTabOpener()
            opener.open(listings, top_n=2)

        # Then: exactly 2 calls
        assert mock_open.call_count == 2, (
            f"Expected 2 tabs opened for top_n=2. Got {mock_open.call_count}"
        )

    def test_tab_count_respects_open_top_n_from_settings_when_cli_absent(
        self,
    ) -> None:
        """
        Given three listings and top_n defaults to 5
        When the browser tab opener is invoked without explicit top_n
        Then the default top_n=5 is used (opens all 3 since fewer available)
        """
        # Given: three listings (fewer than default top_n=5)
        listings = [
            _make_ranked(
                listing=_make_listing(
                    title=f"Role {i}", external_id=str(i),
                    url=f"https://board.com/{i}",
                ),
                final_score=0.9 - (i * 0.1),
            )
            for i in range(3)
        ]

        # When: open without explicit top_n (default=5)
        with patch("jobsearch_rag.export.browser_tabs.webbrowser.open") as mock_open:
            opener = BrowserTabOpener()
            opener.open(listings)

        # Then: all 3 are opened (fewer than default 5)
        assert mock_open.call_count == 3, (
            f"Expected 3 tabs opened (all available < default top_n=5). "
            f"Got {mock_open.call_count}"
        )

    def test_fewer_results_than_n_opens_all_available_without_error(
        self,
    ) -> None:
        """
        Given one listing and top_n=10
        When the browser tab opener is invoked
        Then only 1 tab is opened and no error is raised
        """
        # Given: one listing with top_n much larger
        listing = _make_ranked(
            listing=_make_listing(url="https://board.com/only-one"),
            final_score=0.80,
        )

        # When: open with top_n=10
        with patch("jobsearch_rag.export.browser_tabs.webbrowser.open") as mock_open:
            opener = BrowserTabOpener()
            opener.open([listing], top_n=10)

        # Then: exactly 1 tab opened
        assert mock_open.call_count == 1, (
            f"Expected 1 tab opened when only 1 result available. "
            f"Got {mock_open.call_count}"
        )
        assert mock_open.call_args_list[0].args[0] == "https://board.com/only-one", (
            f"Expected URL 'https://board.com/only-one'. "
            f"Got: {mock_open.call_args_list[0].args[0]}"
        )

    def test_disqualified_roles_are_never_opened(self) -> None:
        """
        Given one qualified and one disqualified listing
        When the browser tab opener is invoked
        Then only the qualified listing's URL is opened
        """
        # Given: one qualified, one disqualified
        qualified = _make_ranked(
            listing=_make_listing(
                title="Good Role", external_id="g", url="https://board.com/good"
            ),
            final_score=0.80,
        )
        disqualified = _make_ranked(
            listing=_make_listing(
                title="Bad Role", external_id="b", url="https://board.com/bad"
            ),
            scores=_make_scores(disqualified=True, disqualifier_reason="Staffing agency"),
            final_score=0.0,
        )

        # When: open with both
        with patch("jobsearch_rag.export.browser_tabs.webbrowser.open") as mock_open:
            opener = BrowserTabOpener()
            opener.open([qualified, disqualified], top_n=10)

        # Then: only the qualified URL is opened
        called_urls = [call.args[0] for call in mock_open.call_args_list]
        assert called_urls == ["https://board.com/good"], (
            f"Expected only qualified URL. Got: {called_urls}"
        )

    def test_failed_tab_open_logs_url_and_continues(self) -> None:
        """
        Given two listings where the first URL fails to open (OSError)
        When the browser tab opener is invoked
        Then the second URL is still opened
        """
        # Given: two listings; first open raises OSError
        r1 = _make_ranked(
            listing=_make_listing(
                title="Failing", external_id="f", url="https://board.com/fail"
            ),
            final_score=0.90,
        )
        r2 = _make_ranked(
            listing=_make_listing(
                title="Succeeding", external_id="s", url="https://board.com/ok"
            ),
            final_score=0.80,
        )

        # When: first open raises OSError
        def side_effect(url: str) -> None:
            if url == "https://board.com/fail":
                raise OSError("Browser launch failed")

        with patch(
            "jobsearch_rag.export.browser_tabs.webbrowser.open",
            side_effect=side_effect,
        ) as mock_open:
            opener = BrowserTabOpener()
            opener.open([r1, r2], top_n=2)

        # Then: both URLs were attempted (2 calls total)
        assert mock_open.call_count == 2, (
            f"Expected 2 open attempts (one fail, one succeed). "
            f"Got {mock_open.call_count}"
        )
        # The second URL was still attempted despite the first failing
        called_urls = [call.args[0] for call in mock_open.call_args_list]
        assert "https://board.com/ok" in called_urls, (
            f"Second URL should have been attempted after first failure. "
            f"Got: {called_urls}"
        )

    def test_zero_scored_results_opens_no_tabs_and_logs_advisory(
        self,
    ) -> None:
        """
        Given an empty list of qualified results (all disqualified)
        When the browser tab opener is invoked
        Then no tabs are opened
        """
        # Given: only disqualified listings
        disqualified = _make_ranked(
            listing=_make_listing(url="https://board.com/bad"),
            scores=_make_scores(disqualified=True, disqualifier_reason="Spam"),
            final_score=0.0,
        )

        # When: open with only disqualified
        with patch("jobsearch_rag.export.browser_tabs.webbrowser.open") as mock_open:
            opener = BrowserTabOpener()
            opener.open([disqualified], top_n=5)

        # Then: no tabs opened
        assert mock_open.call_count == 0, (
            f"Expected 0 tabs opened when all are disqualified. "
            f"Got {mock_open.call_count}"
        )

    def test_open_top_zero_opens_no_tabs_without_error(self) -> None:
        """
        Given a listing and top_n=0
        When the browser tab opener is invoked
        Then no tabs are opened and no error is raised
        """
        # Given: a valid listing with top_n=0
        ranked = _make_ranked(
            listing=_make_listing(url="https://board.com/valid"),
            final_score=0.80,
        )

        # When: open with top_n=0
        with patch("jobsearch_rag.export.browser_tabs.webbrowser.open") as mock_open:
            opener = BrowserTabOpener()
            opener.open([ranked], top_n=0)

        # Then: no tabs opened
        assert mock_open.call_count == 0, (
            f"Expected 0 tabs opened for top_n=0. Got {mock_open.call_count}"
        )

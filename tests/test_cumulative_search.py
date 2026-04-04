"""
Cumulative search — merge-on-export behavior for the search command.

Maps to BDD specs: TestAccumulateMode, TestFreshMode, TestCSVRoundTrip,
TestJDFilePreservation, TestRescoreAccumulatedSet, TestDecisionExclusionAccumulated
"""

from __future__ import annotations

import argparse
import csv
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.cli import handle_search
from jobsearch_rag.export.csv_export import CSVExporter
from jobsearch_rag.export.jd_files import JDFileExporter
from jobsearch_rag.pipeline.ranker import RankedListing, RankSummary
from jobsearch_rag.pipeline.rescorer import load_jd_files
from jobsearch_rag.pipeline.runner import PipelineRunner, RunResult
from jobsearch_rag.rag.scorer import ScoreResult
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from pathlib import Path

# Public API surface (from src/jobsearch_rag/cli):
#   handle_search(args: argparse.Namespace) -> None
#     Accumulate mode (default): merges new results with prior CSV
#     Fresh mode (--fresh): skips merge, replace-on-write
#
# Public API surface (from src/jobsearch_rag/export/csv_export):
#   CSVExporter().export(listings, output_path, *, summary=None) -> None
#
# Public API surface (from src/jobsearch_rag/export/jd_files):
#   JDFileExporter().export(listings, output_dir, *, summary=None) -> list[Path]
#
# Public API surface (from src/jobsearch_rag/pipeline/ranker):
#   RankedListing(listing, scores, final_score, duplicate_boards=[])
#   RankSummary(total_found=0, total_scored=0, total_excluded=0, total_deduplicated=0)
#
# Public API surface (from src/jobsearch_rag/adapters/base):
#   JobListing(board, external_id, title, company, location, url, full_text, ...)
#
# Public API surface (from src/jobsearch_rag/rag/scorer):
#   ScoreResult(fit_score, archetype_score, history_score, disqualified, ...)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(
    board: str = "testboard",
    external_id: str = "ext-1",
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    full_text: str = "A test job description.",
    comp_min: float | None = None,
    comp_max: float | None = None,
) -> JobListing:
    """Create a real JobListing with controlled values."""
    return JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company=company,
        location="Remote",
        url=f"https://example.org/{external_id}",
        full_text=full_text,
        comp_min=comp_min,
        comp_max=comp_max,
    )


def _make_ranked(
    final_score: float = 0.75,
    board: str = "testboard",
    external_id: str = "ext-1",
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    full_text: str = "A test job description.",
    comp_min: float | None = None,
    comp_max: float | None = None,
    disqualified: bool = False,
    disqualifier_reason: str | None = None,
) -> RankedListing:
    """Create a RankedListing with controlled values."""
    listing = _make_listing(
        board=board,
        external_id=external_id,
        title=title,
        company=company,
        full_text=full_text,
        comp_min=comp_min,
        comp_max=comp_max,
    )
    scores = ScoreResult(
        fit_score=0.8,
        archetype_score=0.7,
        history_score=0.5,
        disqualified=disqualified,
        disqualifier_reason=disqualifier_reason,
    )
    return RankedListing(
        listing=listing,
        scores=scores,
        final_score=final_score,
    )


def _write_prior_csv(csv_path: Path, rows: list[RankedListing]) -> None:
    """Write a prior results CSV using the real CSVExporter."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    CSVExporter().export(rows, str(csv_path))


def _write_prior_jd(jd_dir: Path, ranked: RankedListing) -> Path:
    """Write a single JD file and return its path."""
    paths = JDFileExporter().export([ranked], str(jd_dir))
    assert len(paths) == 1, f"Expected 1 JD file, got {len(paths)}"
    return paths[0]


def _setup_search_env(tmp_path: Path, *, open_top_n: int = 0) -> AsyncMock:
    """
    Create config/data files in *tmp_path* and return a mock Ollama client.

    Mirrors ``_setup_index_env`` from test_cli.py but configures
    open_top_n=0 by default (cumulative search tests don't need browser).
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    (config_dir / "settings.toml").write_text(f"""\
resume_path = "{data_dir / "resume.md"}"
archetypes_path = "{config_dir / "role_archetypes.toml"}"
global_rubric_path = "{config_dir / "global_rubric.toml"}"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
max_pages = 2
headless = true

[scoring]

[ollama]

[output]
output_dir = "{output_dir}"
open_top_n = {open_top_n}

[chroma]
persist_dir = "{tmp_path / "chroma"}"
""")

    (config_dir / "role_archetypes.toml").write_text("""\
[[archetypes]]
name = "Test Archetype"
description = "A test archetype for indexing."
signals_positive = ["positive signal"]
signals_negative = ["negative signal"]
""")

    (config_dir / "global_rubric.toml").write_text("""\
[[dimensions]]
name = "Test Dimension"
signals_positive = ["good indicator"]
signals_negative = ["bad indicator"]
""")

    (data_dir / "resume.md").write_text("""\
## Summary

Test resume content for indexing.
""")

    mock_client = AsyncMock()
    model_embed = MagicMock()
    model_embed.model = "nomic-embed-text"
    model_llm = MagicMock()
    model_llm.model = "mistral:7b"
    mock_client.list.return_value = MagicMock(models=[model_embed, model_llm])
    mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]])

    return mock_client


def _seed_decision(
    tmp_path: Path,
    *,
    job_id: str,
    verdict: str = "skip",
    board: str = "testboard",
    title: str = "Decided Job",
    company: str = "Acme Corp",
) -> None:
    """Pre-populate the ChromaDB decisions collection with a test record."""
    store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    store.get_or_create_collection("decisions")
    store.add_documents(
        collection_name="decisions",
        ids=[f"decision-{job_id}"],
        documents=["Full JD text for a decided listing."],
        embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
        metadatas=[
            {
                "job_id": job_id,
                "verdict": verdict,
                "board": board,
                "title": title,
                "company": company,
            }
        ],
    )


# ---------------------------------------------------------------------------
# B1 — Accumulate Mode (Default)
# ---------------------------------------------------------------------------


class TestAccumulateMode:
    """
    REQUIREMENT: By default, search merges new results with prior export
    data instead of replacing it.

    WHO: Operator running daily searches that accumulate over the week
    WHAT: (1) New listings not in prior CSV are appended to the merged set
          (2) New listings matching a prior external_id replace the prior row
              with updated scores
          (3) Prior-only listings (not in the current run) are preserved
          (4) The merged list is sorted by final_score descending
          (5) Markdown summary reflects the merged set totals
    WHY: Decouples search frequency from review cadence — run searches
         unattended during the week, review accumulated results on the weekend

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama network I/O),
               PipelineRunner.run (browser/scraping I/O)
        Real:  load_settings, CSV merge logic, file exports via tmp_path
        Never: Patch exporters or ranker
    """

    def test_new_listings_are_appended_to_prior_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with one listing and a new run with a different listing
        When handle_search runs in accumulate mode
        Then the exported CSV contains both prior and new listings
        """
        # Given: prior CSV with listing A
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.80, external_id="prior-1", title="Prior Job")
        _write_prior_csv(output_dir / "results.csv", [prior])

        new = _make_ranked(final_score=0.90, external_id="new-1", title="New Job")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode (default)
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: exported CSV contains both listings
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        ids = {row["external_id"] for row in rows}
        assert ids == {"prior-1", "new-1"}, (
            f"Expected both external_ids in exported CSV, got {ids}"
        )

    def test_matching_external_id_replaces_prior_with_updated_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with external_id 'X' scoring 0.60
        When handle_search returns external_id 'X' scoring 0.85
        Then the exported CSV contains 'X' once with the updated score
        """
        # Given: prior CSV with listing scoring 0.60
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.60, external_id="shared-1", title="Shared Job")
        _write_prior_csv(output_dir / "results.csv", [prior])

        # And: new run with same external_id but higher score
        new = _make_ranked(final_score=0.85, external_id="shared-1", title="Shared Job Updated")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: CSV has one copy with the new score
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        shared_rows = [r for r in rows if r["external_id"] == "shared-1"]
        assert len(shared_rows) == 1, f"Expected 1 row for shared-1, got {len(shared_rows)}"
        assert float(shared_rows[0]["final_score"]) == pytest.approx(0.85, abs=1e-4), (
            f"Expected score ~0.85, got {shared_rows[0]['final_score']}"
        )

    def test_prior_only_listings_are_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with listings A and B
        When handle_search returns only listing C
        Then the exported CSV contains A, B, and C
        """
        # Given: two prior listings
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior_a = _make_ranked(final_score=0.70, external_id="prior-a", title="Job A")
        prior_b = _make_ranked(final_score=0.65, external_id="prior-b", title="Job B")
        _write_prior_csv(output_dir / "results.csv", [prior_a, prior_b])

        new_c = _make_ranked(final_score=0.90, external_id="new-c", title="Job C")
        result = RunResult(
            ranked_listings=[new_c],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: all three listings in the exported CSV
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        ids = {row["external_id"] for row in rows}
        assert ids == {"prior-a", "prior-b", "new-c"}, (
            f"Expected all three external_ids, got {ids}"
        )

    def test_merged_list_is_sorted_by_final_score_descending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with a low-scoring listing
        When handle_search returns a high-scoring listing
        Then the exported CSV is ordered by final_score descending
        """
        # Given: prior listing scoring 0.50
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.50, external_id="low-1", title="Low Score")
        _write_prior_csv(output_dir / "results.csv", [prior])

        new = _make_ranked(final_score=0.95, external_id="high-1", title="High Score")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: CSV is sorted descending by final_score
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        scores = [float(row["final_score"]) for row in rows]
        assert scores == sorted(scores, reverse=True), f"Expected descending sort, got {scores}"
        assert len(scores) >= 2, f"Expected at least 2 rows, got {len(scores)}"

    def test_markdown_summary_reflects_merged_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with one listing
        When handle_search returns a different listing in accumulate mode
        Then the markdown summary table contains both listings
        """
        # Given: prior CSV with listing A
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.70, external_id="md-prior", title="Prior MD Job")
        _write_prior_csv(output_dir / "results.csv", [prior])

        new = _make_ranked(final_score=0.90, external_id="md-new", title="New MD Job")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: markdown file contains both listings in its table
        md_path = output_dir / "results.md"
        md_text = md_path.read_text()
        assert "Prior MD Job" in md_text, "Expected prior listing in markdown summary"
        assert "New MD Job" in md_text, "Expected new listing in markdown summary"


# ---------------------------------------------------------------------------
# B2 — Fresh Mode Override
# ---------------------------------------------------------------------------


class TestFreshMode:
    """
    REQUIREMENT: The --fresh flag restores replace-on-write behavior.

    WHO: Operator starting a clean search after a config change
    WHAT: (1) When --fresh is set, handle_search skips merge entirely
          (2) Exporters receive only the current run's results
          (3) Stale JD file cleanup proceeds normally (Phase 6h behavior)
    WHY: After re-indexing or weight changes, prior scores are stale —
         the operator needs a way to start fresh

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama network I/O),
               PipelineRunner.run (browser/scraping I/O)
        Real:  load_settings, file exports via tmp_path, JD file cleanup
        Never: Patch exporters or merge logic
    """

    def test_fresh_flag_ignores_prior_csv(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given a prior CSV with listing A and --fresh is set
        When handle_search runs with a new listing B
        Then the exported CSV contains only listing B
        """
        # Given: set up environment and seed a prior CSV
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.70, external_id="prior-1", title="Prior Job")
        _write_prior_csv(output_dir / "results.csv", [prior])

        new = _make_ranked(final_score=0.90, external_id="new-1", title="New Job")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search runs with --fresh
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=True,
                )
            )

        # Then: CSV contains only the new listing
        csv_path = output_dir / "results.csv"
        assert csv_path.exists(), "CSV file should exist"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        ids = {row["external_id"] for row in rows}
        assert ids == {"new-1"}, f"Expected only 'new-1' in fresh CSV, got {ids}"

    def test_fresh_flag_removes_stale_jd_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given a prior JD file for listing A and --fresh is set
        When handle_search runs with listing B only
        Then the JD file for listing A is removed
        """
        # Given: set up environment and seed a prior JD file
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"
        jd_dir = output_dir / "jds"

        prior = _make_ranked(final_score=0.70, external_id="prior-1", title="Prior Job")
        prior_jd_path = _write_prior_jd(jd_dir, prior)
        assert prior_jd_path.exists(), "Prior JD file should exist before search"

        new = _make_ranked(final_score=0.90, external_id="new-1", title="New Job")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search runs with --fresh
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=True,
                )
            )

        # Then: prior JD file is gone, new JD file exists
        assert not prior_jd_path.exists(), "Prior JD file should be removed in fresh mode"
        jd_files = list(jd_dir.glob("*.md"))
        assert len(jd_files) == 1, f"Expected 1 JD file, got {len(jd_files)}"
        assert "new-1" in jd_files[0].name, (
            f"Expected new-1 in JD filename, got {jd_files[0].name}"
        )

    def test_fresh_exporters_receive_only_current_run_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given a prior CSV with listing A and --fresh is set
        When handle_search runs with listing B only
        Then the markdown summary contains only listing B
        """
        # Given: set up environment and seed a prior CSV
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.70, external_id="prior-1", title="Prior Fresh")
        _write_prior_csv(output_dir / "results.csv", [prior])

        new = _make_ranked(final_score=0.90, external_id="new-1", title="New Fresh")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search runs with --fresh
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=True,
                )
            )

        # Then: markdown contains only the new listing
        md_path = output_dir / "results.md"
        md_text = md_path.read_text()
        assert "New Fresh" in md_text, "Expected new listing in fresh markdown summary"
        assert "Prior Fresh" not in md_text, (
            "Expected prior listing absent from fresh markdown summary"
        )


# ---------------------------------------------------------------------------
# B3 — CSV Round-Trip Fidelity
# ---------------------------------------------------------------------------


class TestCSVRoundTrip:
    """
    REQUIREMENT: Prior CSV rows round-trip cleanly through load -> merge -> export.

    WHO: The merge function reconstructing RankedListing from CSV rows
    WHAT: (1) All 17 CSV columns are parsed back into RankedListing with
              correct types
          (2) external_id is the merge key (not title, not URL)
          (3) No duplicate external_id rows appear after merge
          (4) Float precision is preserved (4 decimal places)
          (5) Header row appears exactly once after merge
    WHY: A broken round-trip silently corrupts accumulated data

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama network I/O),
               PipelineRunner.run (browser/scraping I/O)
        Real:  load_settings, CSV merge logic, CSVExporter, CSV I/O via tmp_path
        Never: Patch CSV writer or reader
    """

    def test_all_csv_columns_round_trip_with_correct_types(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with fully populated columns
        When handle_search runs in accumulate mode with a different listing
        Then the prior listing's columns are preserved in the exported CSV
        """
        # Given: prior CSV with all fields populated
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        original = _make_ranked(
            final_score=0.8765,
            external_id="rt-1",
            title="Round Trip Job",
            company="RT Corp",
            comp_min=120000.0,
            comp_max=180000.0,
        )
        _write_prior_csv(output_dir / "results.csv", [original])

        # And: a new listing to trigger export
        new = _make_ranked(final_score=0.50, external_id="other-1", title="Other Job")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: prior listing's columns are preserved
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        prior_rows = [r for r in rows if r["external_id"] == "rt-1"]
        assert len(prior_rows) == 1, f"Expected 1 row for rt-1, got {len(prior_rows)}"
        r = prior_rows[0]
        assert r["title"] == "Round Trip Job", (
            f"Expected title 'Round Trip Job', got {r['title']!r}"
        )
        assert r["company"] == "RT Corp", f"Expected company 'RT Corp', got {r['company']!r}"
        assert r["board"] == "testboard", f"Expected board 'testboard', got {r['board']!r}"
        assert r["location"] == "Remote", f"Expected location 'Remote', got {r['location']!r}"
        assert float(r["final_score"]) == pytest.approx(0.8765, abs=1e-4), (
            f"Expected final_score ~0.8765, got {r['final_score']}"
        )
        assert float(r["fit_score"]) == pytest.approx(0.8, abs=1e-4), (
            f"Expected fit_score ~0.8, got {r['fit_score']}"
        )
        assert float(r["archetype_score"]) == pytest.approx(0.7, abs=1e-4), (
            f"Expected archetype_score ~0.7, got {r['archetype_score']}"
        )
        assert float(r["history_score"]) == pytest.approx(0.5, abs=1e-4), (
            f"Expected history_score ~0.5, got {r['history_score']}"
        )

    def test_external_id_is_the_merge_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with external_id 'X' and title 'Original'
        When handle_search returns external_id 'X' with title 'Updated'
        Then the exported CSV has one entry with the new title
        """
        # Given: prior listing
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.60, external_id="merge-key-1", title="Original Title")
        _write_prior_csv(output_dir / "results.csv", [prior])

        # And: new listing with same external_id, different title
        new = _make_ranked(final_score=0.80, external_id="merge-key-1", title="Updated Title")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: one entry with the new title
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        key_rows = [r for r in rows if r["external_id"] == "merge-key-1"]
        assert len(key_rows) == 1, f"Expected 1 entry, got {len(key_rows)}"
        assert key_rows[0]["title"] == "Updated Title", (
            f"Expected 'Updated Title', got {key_rows[0]['title']!r}"
        )

    def test_no_duplicate_external_ids_after_merge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with external_id 'dup-1'
        When handle_search returns external_id 'dup-1'
        Then the exported CSV has exactly one 'dup-1' row
        """
        # Given: prior with dup-1
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.60, external_id="dup-1", title="Dup Job")
        _write_prior_csv(output_dir / "results.csv", [prior])

        new = _make_ranked(final_score=0.80, external_id="dup-1", title="Dup Job New")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: exactly one dup-1 row
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        dup_count = sum(1 for r in rows if r["external_id"] == "dup-1")
        assert dup_count == 1, f"Expected exactly one 'dup-1', got {dup_count}"

    def test_float_precision_preserved_through_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with 4-decimal-place scores
        When handle_search runs in accumulate mode
        Then scores are preserved to 4 decimal places in the exported CSV
        """
        # Given: listing with precise scores
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        original = _make_ranked(final_score=0.7531, external_id="prec-1")
        _write_prior_csv(output_dir / "results.csv", [original])

        new = _make_ranked(final_score=0.50, external_id="other-prec", title="Other Prec")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: precision preserved
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        prec_rows = [r for r in rows if r["external_id"] == "prec-1"]
        assert len(prec_rows) == 1, f"Expected 1 row for prec-1, got {len(prec_rows)}"
        assert float(prec_rows[0]["final_score"]) == pytest.approx(0.7531, abs=1e-4), (
            f"Expected 0.7531, got {prec_rows[0]['final_score']}"
        )

    def test_header_row_appears_exactly_once_after_merge_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with one listing
        When handle_search runs in accumulate mode
        Then the exported CSV has exactly one header row
        """
        # Given: prior CSV
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior = _make_ranked(final_score=0.70, external_id="hdr-1")
        _write_prior_csv(output_dir / "results.csv", [prior])

        new = _make_ranked(final_score=0.90, external_id="hdr-new", title="Header Test")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: exactly one header row
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            lines = f.readlines()
        header_count = sum(1 for line in lines if line.startswith("title,"))
        assert header_count == 1, f"Expected exactly 1 header row, got {header_count}"


# ---------------------------------------------------------------------------
# B4 — JD File Preservation in Accumulate Mode
# ---------------------------------------------------------------------------


class TestJDFilePreservation:
    """
    REQUIREMENT: JD files from prior runs are preserved when not in the
    current result set (accumulate mode).

    WHO: The export step in accumulate mode
    WHAT: (1) JD files for listings in the current run are written/overwritten
          (2) JD files NOT in the current run are NOT deleted in accumulate mode
    WHY: The accumulated JD files are the source of truth for rescore and review

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama network I/O),
               PipelineRunner.run (browser/scraping I/O)
        Real:  load_settings, file exports via tmp_path, JD file I/O
        Never: Patch JDFileExporter
    """

    def test_current_run_jd_files_are_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given no prior JD files exist
        When handle_search runs in accumulate mode with one listing
        Then a JD file is created for that listing
        """
        # Given: clean environment
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"
        jd_dir = output_dir / "jds"

        new = _make_ranked(final_score=0.85, external_id="jd-1", title="JD Job")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in default accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: JD file exists for the listing
        jd_files = list(jd_dir.glob("*.md"))
        assert len(jd_files) >= 1, f"Expected at least 1 JD file, got {len(jd_files)}"
        assert any("jd-1" in f.name for f in jd_files), (
            f"Expected JD file with 'jd-1' in name, got {[f.name for f in jd_files]}"
        )

    def test_prior_jd_files_preserved_in_accumulate_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given a prior JD file for listing A exists
        When handle_search runs in accumulate mode with listing B only
        Then the JD file for listing A is preserved
        """
        # Given: seed a prior JD file
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"
        jd_dir = output_dir / "jds"

        prior = _make_ranked(final_score=0.70, external_id="prior-jd", title="Prior JD")
        prior_jd_path = _write_prior_jd(jd_dir, prior)
        assert prior_jd_path.exists(), "Prior JD file should exist"

        new = _make_ranked(final_score=0.90, external_id="new-jd", title="New JD")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode (default)
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: prior JD file is preserved, new JD file exists
        assert prior_jd_path.exists(), "Prior JD file should be preserved in accumulate mode"
        jd_files = list(jd_dir.glob("*.md"))
        assert len(jd_files) == 2, f"Expected 2 JD files, got {len(jd_files)}"


# ---------------------------------------------------------------------------
# B5 — Rescore Handles Accumulated Set
# ---------------------------------------------------------------------------


class TestRescoreAccumulatedSet:
    """
    REQUIREMENT: rescore naturally operates on all JD files regardless of
    accumulation.

    WHO: Operator re-scoring after config changes
    WHAT: (1) load_jd_files reads all .md files in jds/ — existing behavior
              works unchanged over accumulated files from multiple runs
    WHY: Validate that the existing rescore path handles accumulated files

    MOCK BOUNDARY:
        Mock:  nothing — this class tests pure filesystem I/O via tmp_path
        Real:  load_jd_files, JDFileExporter, tmp_path filesystem
        Never: Patch load_jd_files internals
    """

    def test_load_jd_files_reads_all_accumulated_files(self, tmp_path: Path) -> None:
        """
        Given JD files from two separate export runs exist in the same directory
        When load_jd_files reads the directory
        Then all files are loaded regardless of which run created them
        """
        # Given: write JD files as if from two separate runs
        jd_dir = tmp_path / "jds"
        run1 = _make_ranked(
            final_score=0.70,
            external_id="run1-1",
            title="Run1 Job",
            company="Run1 Corp",
        )
        run2 = _make_ranked(
            final_score=0.85,
            external_id="run2-1",
            title="Run2 Job",
            company="Run2 Corp",
        )
        JDFileExporter().export([run1, run2], str(jd_dir))

        # When: load_jd_files reads the directory
        loaded = load_jd_files(jd_dir)

        # Then: both files are loaded
        ids = {listing.external_id for listing in loaded}
        assert "run1-1" in ids, f"Expected 'run1-1' in loaded IDs, got {ids}"
        assert "run2-1" in ids, f"Expected 'run2-1' in loaded IDs, got {ids}"


# ---------------------------------------------------------------------------
# B6 — Decision Exclusion in Accumulate Mode
# ---------------------------------------------------------------------------


class TestDecisionExclusionAccumulated:
    """
    REQUIREMENT: Decided listings are excluded from the merged CSV export
    even if their JD files are preserved.

    WHO: Operator who reviewed some listings between search runs
    WHAT: (1) The merge step filters decided listings from the merged set
              before passing to exporters
          (2) JD files for decided listings are preserved (needed for eval)
    WHY: Prevents already-reviewed listings from cluttering the ranked output

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama network I/O),
               PipelineRunner.run (browser/scraping I/O)
        Real:  load_settings, CSV merge logic, DecisionRecorder via ChromaDB,
               CSV/JD I/O via tmp_path
        Never: Patch the decision lookup
    """

    def test_decided_listings_excluded_from_merged_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Given a prior CSV with listing A (decided) and listing B (undecided)
        When handle_search runs in accumulate mode
        Then the exported CSV does not contain listing A
        """
        # Given: two prior listings, one of which was decided
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"

        prior_a = _make_ranked(final_score=0.70, external_id="decided-1", title="Decided Job")
        prior_b = _make_ranked(final_score=0.65, external_id="undecided-1", title="Undecided Job")
        _write_prior_csv(output_dir / "results.csv", [prior_a, prior_b])

        # And: seed a decision for listing A
        _seed_decision(tmp_path, job_id="decided-1", title="Decided Job")

        new = _make_ranked(final_score=0.90, external_id="new-1", title="New Job")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: decided listing excluded, undecided listing present
        csv_path = output_dir / "results.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        ids = {row["external_id"] for row in rows}
        assert "decided-1" not in ids, f"Expected decided listing excluded from CSV, got {ids}"
        assert "undecided-1" in ids, f"Expected undecided listing in CSV, got {ids}"

    def test_jd_files_for_decided_listings_are_preserved(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given a prior JD file for a decided listing
        When handle_search runs in accumulate mode
        Then the JD file for the decided listing is preserved
        """
        # Given: seed a JD file for a decided listing
        mock_client = _setup_search_env(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "output"
        jd_dir = output_dir / "jds"

        decided = _make_ranked(final_score=0.70, external_id="decided-jd", title="Decided JD")
        decided_jd_path = _write_prior_jd(jd_dir, decided)
        assert decided_jd_path.exists(), "Decided JD file should exist"

        # And: seed the prior CSV (includes the decided listing)
        _write_prior_csv(output_dir / "results.csv", [decided])

        new = _make_ranked(final_score=0.90, external_id="new-jd-2", title="New JD 2")
        result = RunResult(
            ranked_listings=[new],
            summary=RankSummary(total_found=1, total_scored=1),
            boards_searched=["testboard"],
        )

        # When: handle_search in accumulate mode
        with (
            patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client),
            patch.object(PipelineRunner, "run", new_callable=AsyncMock, return_value=result),
            patch("webbrowser.open"),
        ):
            handle_search(
                argparse.Namespace(
                    board=None,
                    overnight=False,
                    open_top=0,
                    force_rescore=False,
                    fresh=False,
                )
            )

        # Then: decided JD file is preserved
        assert decided_jd_path.exists(), "JD file for decided listing should be preserved"

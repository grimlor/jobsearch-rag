"""Rescore pipeline tests — JD file loading and re-scoring workflow.

Maps to BDD specs: TestJdFileLoading, TestRescoreWorkflow
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.pipeline.ranker import Ranker
from jobsearch_rag.pipeline.rescorer import (
    Rescorer,
    RescoreResult,
    load_jd_files,
)
from jobsearch_rag.rag.scorer import ScoreResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_JD_CONTENT = textwrap.dedent("""\
    # Staff Platform Architect

    **Company:** Acme Corp
    **Location:** Remote
    **Board:** ziprecruiter
    **URL:** https://example.org/job/42

    ## Score

    - **Rank:** #1
    - **Final Score:** 0.85
    - **Fit Score:** 0.80
    - **Archetype Score:** 0.90
    - **History Score:** 0.70
    - **Comp Score:** 0.50
    - **Negative Score:** 0.10

    ## Job Description

    We are looking for a Staff Platform Architect to lead our cloud
    infrastructure team.  You will design distributed systems and
    mentor senior engineers.  Experience with Kubernetes and Terraform
    is required.
""")


@pytest.fixture()
def jd_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a sample JD file."""
    d = tmp_path / "jds"
    d.mkdir()
    (d / "001_acme-corp_staff-platform-architect.md").write_text(_SAMPLE_JD_CONTENT)
    return d


# ---------------------------------------------------------------------------
# TestJdFileLoading
# ---------------------------------------------------------------------------


class TestJdFileLoading:
    """REQUIREMENT: JD files from output/jds/ are loaded back into JobListing objects.

    WHO: The rescore pipeline; the operator running ``rescore`` after tuning archetypes
    WHAT: ``load_jd_files()`` reads markdown JD files, parses metadata headers
          (title, company, board, URL, location) and JD body, and returns
          JobListing objects ready for re-scoring; missing fields use sensible
          defaults; files without a JD body are skipped; non-existent directories
          return an empty list
    WHY: Without correct JD loading, rescoring would fail silently or produce
         garbled results — the operator would have to re-run full browser sessions
    """

    def test_loads_listing_from_valid_jd_file(self, jd_dir: Path) -> None:
        """A well-formed JD file is loaded into a JobListing with all metadata."""
        listings = load_jd_files(jd_dir)
        assert len(listings) == 1
        listing = listings[0]
        assert listing.title == "Staff Platform Architect"
        assert listing.company == "Acme Corp"
        assert listing.board == "ziprecruiter"
        assert listing.url == "https://example.org/job/42"
        assert listing.location == "Remote"

    def test_listing_full_text_contains_jd_body(self, jd_dir: Path) -> None:
        """The loaded listing's full_text contains the JD body text."""
        listings = load_jd_files(jd_dir)
        assert "Staff Platform Architect to lead" in listings[0].full_text
        assert "Kubernetes and Terraform" in listings[0].full_text

    def test_external_id_derived_from_url(self, jd_dir: Path) -> None:
        """The external_id is extracted from the last URL path segment."""
        listings = load_jd_files(jd_dir)
        assert listings[0].external_id == "42"

    def test_nonexistent_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """A non-existent directory returns an empty list, not an error."""
        listings = load_jd_files(tmp_path / "does-not-exist")
        assert listings == []

    def test_file_without_jd_body_is_skipped(self, tmp_path: Path) -> None:
        """A JD file missing the '## Job Description' section is silently skipped."""
        d = tmp_path / "jds"
        d.mkdir()
        (d / "001_no-body.md").write_text("# Title\n\n**Company:** Co\n")
        listings = load_jd_files(d)
        assert listings == []

    def test_missing_metadata_uses_defaults(self, tmp_path: Path) -> None:
        """A JD file with only a body still loads with default metadata values."""
        d = tmp_path / "jds"
        d.mkdir()
        content = textwrap.dedent("""\
            ## Job Description

            Some job description text here.
        """)
        (d / "minimal.md").write_text(content)
        listings = load_jd_files(d)
        assert len(listings) == 1
        assert listings[0].company == "Unknown"
        assert listings[0].board == "unknown"

    def test_multiple_jd_files_loaded_in_sorted_order(self, tmp_path: Path) -> None:
        """Multiple JD files are loaded in filename-sorted order."""
        d = tmp_path / "jds"
        d.mkdir()
        for i, (title, company) in enumerate(
            [("Role B", "Beta"), ("Role A", "Alpha"), ("Role C", "Gamma")], 1
        ):
            content = textwrap.dedent(f"""\
                # {title}

                **Company:** {company}
                **Board:** test

                ## Job Description

                Description for {title}.
            """)
            (d / f"{i:03d}_{company.lower()}_{title.lower().replace(' ', '-')}.md").write_text(
                content
            )
        listings = load_jd_files(d)
        assert len(listings) == 3
        # Sorted by filename: 001, 002, 003
        assert listings[0].title == "Role B"
        assert listings[1].title == "Role A"
        assert listings[2].title == "Role C"


# ---------------------------------------------------------------------------
# TestRescoreWorkflow
# ---------------------------------------------------------------------------


class TestRescoreWorkflow:
    """REQUIREMENT: The rescore pipeline re-scores JDs through updated RAG collections.

    WHO: The operator iterating on archetype tuning or negative signal refinement
    WHAT: The Rescorer loads JD files, scores each through the current scorer,
          ranks results through the ranker, and returns a RescoreResult with
          ranked listings and summary statistics; failures are counted and
          non-fatal; an empty directory produces an empty result
    WHY: Without rescoring, the operator would need to re-run full browser
         sessions after every archetype or rubric tweak — a 20-minute wait
         instead of seconds
    """

    def _make_mock_scorer(self, scores: dict[str, Any] | None = None) -> MagicMock:
        """Create a mock Scorer that returns controlled ScoreResult values."""
        default = ScoreResult(
            fit_score=0.8,
            archetype_score=0.7,
            history_score=0.5,
            negative_score=0.1,
            disqualified=False,
        )
        mock = MagicMock()
        mock.score = AsyncMock(return_value=scores or default)
        return mock

    def _make_ranker(self) -> Ranker:
        """Create a real Ranker with default weights."""
        return Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            comp_weight=0.0,
            negative_weight=0.4,
            min_score_threshold=0.0,
        )

    @pytest.mark.asyncio()
    async def test_rescore_returns_ranked_listings(self, jd_dir: Path) -> None:
        """Rescoring a directory with JD files returns ranked results."""
        scorer = self._make_mock_scorer()
        ranker = self._make_ranker()
        rescorer = Rescorer(scorer=scorer, ranker=ranker)
        result = await rescorer.rescore(str(jd_dir))
        assert result.total_loaded == 1
        assert len(result.ranked_listings) == 1
        assert result.failed_listings == 0

    @pytest.mark.asyncio()
    async def test_rescore_empty_directory_returns_empty_result(self, tmp_path: Path) -> None:
        """Rescoring an empty directory returns an empty RescoreResult."""
        scorer = self._make_mock_scorer()
        ranker = self._make_ranker()
        rescorer = Rescorer(scorer=scorer, ranker=ranker)
        result = await rescorer.rescore(str(tmp_path / "empty"))
        assert result.total_loaded == 0
        assert len(result.ranked_listings) == 0

    @pytest.mark.asyncio()
    async def test_rescore_counts_failed_listings(self, jd_dir: Path) -> None:
        """Scoring failures are counted in failed_listings and don't crash the run."""
        scorer = MagicMock()
        scorer.score = AsyncMock(
            side_effect=ActionableError.connection(
                "ollama", "localhost:11434", "Connection refused"
            )
        )
        ranker = self._make_ranker()
        rescorer = Rescorer(scorer=scorer, ranker=ranker)
        result = await rescorer.rescore(str(jd_dir))
        assert result.total_loaded == 1
        assert result.failed_listings == 1
        assert len(result.ranked_listings) == 0

    @pytest.mark.asyncio()
    async def test_rescore_result_has_default_fields(self) -> None:
        """A fresh RescoreResult has sensible defaults."""
        result = RescoreResult()
        assert result.total_loaded == 0
        assert result.failed_listings == 0
        assert result.ranked_listings == []

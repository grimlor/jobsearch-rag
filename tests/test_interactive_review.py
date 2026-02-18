"""Interactive review tests — batch decision recording on undecided listings.

Maps to BDD spec: TestInteractiveReview

Tests verify that the ``review`` command loads latest search results in
ranked order, skips already-decided listings, displays scores and
component breakdowns, records verdicts immediately via DecisionRecorder,
shows progress, and preserves verdicts on quit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.pipeline.ranker import RankedListing
from jobsearch_rag.pipeline.review import ReviewSession
from jobsearch_rag.rag.scorer import ScoreResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ranked(
    *,
    title: str = "Solutions Architect",
    company: str = "Acme Corp",
    board: str = "ziprecruiter",
    external_id: str = "zr-1",
    final_score: float = 0.82,
    fit: float = 0.85,
    archetype: float = 0.90,
    history: float = 0.50,
    comp: float = 0.60,
    comp_min: float | None = 180_000,
    comp_max: float | None = 250_000,
    url: str = "https://www.ziprecruiter.com/jobs/test-1",
) -> RankedListing:
    listing = JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company=company,
        location="Remote",
        url=url,
        full_text="Full JD text for testing." * 10,
        comp_min=comp_min,
        comp_max=comp_max,
    )
    scores = ScoreResult(
        fit_score=fit,
        archetype_score=archetype,
        history_score=history,
        disqualified=False,
        comp_score=comp,
    )
    return RankedListing(listing=listing, scores=scores, final_score=final_score)


def _make_recorder(decided_ids: set[str] | None = None) -> MagicMock:
    """Create a mock DecisionRecorder that knows about decided_ids."""
    recorder = MagicMock()
    recorder.record = AsyncMock()
    decided = decided_ids or set()
    recorder.get_decision = MagicMock(
        side_effect=lambda jid: {"verdict": "yes"} if jid in decided else None
    )
    return recorder


class TestInteractiveReview:
    """The operator can review and decide on all undecided listings
    in a single interactive session rather than one-at-a-time CLI calls."""

    def test_review_loads_latest_results_in_ranked_order(self) -> None:
        """Results are presented in descending score order — the operator
        sees the best matches first."""
        high = _make_ranked(title="High Score", final_score=0.95, external_id="h1")
        low = _make_ranked(title="Low Score", final_score=0.55, external_id="l1")
        mid = _make_ranked(title="Mid Score", final_score=0.75, external_id="m1")

        session = ReviewSession(
            ranked_listings=[low, high, mid],
            recorder=_make_recorder(),
        )
        # Undecided should be in ranked (descending) order
        undecided = session.undecided_listings()
        assert [r.listing.title for r in undecided] == [
            "High Score", "Mid Score", "Low Score"
        ]

    def test_already_decided_listings_are_excluded(self) -> None:
        """Listings with an existing decision are skipped so the operator
        only sees roles that still need a verdict."""
        r1 = _make_ranked(external_id="decided-1")
        r2 = _make_ranked(external_id="undecided-1")

        recorder = _make_recorder(decided_ids={"decided-1"})
        session = ReviewSession(ranked_listings=[r1, r2], recorder=recorder)

        undecided = session.undecided_listings()
        assert len(undecided) == 1
        assert undecided[0].listing.external_id == "undecided-1"

    def test_listing_display_shows_rank_title_company_score(self) -> None:
        """Each listing display includes rank position, title, company,
        and final score so the operator has the essential context."""
        ranked = _make_ranked(
            title="Staff Architect", company="TechCo", final_score=0.88
        )
        session = ReviewSession(ranked_listings=[ranked], recorder=_make_recorder())
        output = session.format_listing(ranked, rank=1, total=5)

        assert "1" in output  # rank
        assert "Staff Architect" in output
        assert "TechCo" in output
        assert "0.88" in output

    def test_listing_display_shows_component_score_breakdown(self) -> None:
        """The display includes individual component scores so the operator
        can see why a listing ranked where it did."""
        ranked = _make_ranked(fit=0.85, archetype=0.90, history=0.50, comp=0.60)
        session = ReviewSession(ranked_listings=[ranked], recorder=_make_recorder())
        output = session.format_listing(ranked, rank=1, total=1)

        assert "0.85" in output  # fit
        assert "0.90" in output  # archetype
        assert "0.50" in output  # history
        assert "0.60" in output  # comp

    def test_listing_display_shows_comp_range_when_available(self) -> None:
        """Compensation range is shown when the comp parser found data,
        helping the operator assess financial fit at a glance."""
        ranked = _make_ranked(comp_min=180_000, comp_max=250_000)
        session = ReviewSession(ranked_listings=[ranked], recorder=_make_recorder())
        output = session.format_listing(ranked, rank=1, total=1)

        assert "180" in output  # comp_min (may be formatted)
        assert "250" in output  # comp_max

    @pytest.mark.asyncio
    async def test_yes_verdict_records_via_decision_recorder(self) -> None:
        """Entering 'y' records a 'yes' verdict through DecisionRecorder."""
        ranked = _make_ranked(external_id="job-1")
        recorder = _make_recorder()
        session = ReviewSession(ranked_listings=[ranked], recorder=recorder)

        await session.record_verdict(ranked, "y")
        recorder.record.assert_called_once()
        call_kwargs = recorder.record.call_args.kwargs
        assert call_kwargs["verdict"] == "yes"
        assert call_kwargs["job_id"] == "job-1"

    @pytest.mark.asyncio
    async def test_no_verdict_records_via_decision_recorder(self) -> None:
        """Entering 'n' records a 'no' verdict."""
        ranked = _make_ranked(external_id="job-2")
        recorder = _make_recorder()
        session = ReviewSession(ranked_listings=[ranked], recorder=recorder)

        await session.record_verdict(ranked, "n")
        call_kwargs = recorder.record.call_args.kwargs
        assert call_kwargs["verdict"] == "no"

    @pytest.mark.asyncio
    async def test_maybe_verdict_records_via_decision_recorder(self) -> None:
        """Entering 'm' records a 'maybe' verdict."""
        ranked = _make_ranked(external_id="job-3")
        recorder = _make_recorder()
        session = ReviewSession(ranked_listings=[ranked], recorder=recorder)

        await session.record_verdict(ranked, "m")
        call_kwargs = recorder.record.call_args.kwargs
        assert call_kwargs["verdict"] == "maybe"

    def test_skip_advances_without_recording(self) -> None:
        """Entering 's' advances to the next listing without recording
        any verdict — the listing remains undecided for future review."""
        ranked = _make_ranked(external_id="skip-me")
        recorder = _make_recorder()
        session = ReviewSession(ranked_listings=[ranked], recorder=recorder)

        # 's' should not trigger record
        result = session.should_record("s")
        assert result is False

    @pytest.mark.asyncio
    async def test_open_launches_jd_file_in_system_viewer(self) -> None:
        """Entering 'o' opens the JD file in the system default viewer
        so the operator can read the full description."""
        ranked = _make_ranked(external_id="open-me")
        session = ReviewSession(
            ranked_listings=[ranked],
            recorder=_make_recorder(),
            jd_dir="output/jds",
        )

        with patch("jobsearch_rag.pipeline.review.webbrowser.open") as mock_open:
            session.open_listing(ranked)
            mock_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_quit_preserves_all_previously_recorded_verdicts(self) -> None:
        """Quitting mid-review does not roll back or lose any verdicts
        that were recorded before the quit — each verdict is persisted
        immediately when entered."""
        r1 = _make_ranked(external_id="keep-1")
        r2 = _make_ranked(external_id="keep-2")
        r3 = _make_ranked(external_id="quit-before")
        recorder = _make_recorder()
        session = ReviewSession(
            ranked_listings=[r1, r2, r3], recorder=recorder
        )

        # Record two verdicts then simulate quit
        await session.record_verdict(r1, "y")
        await session.record_verdict(r2, "n")
        # Don't record r3 — simulates quitting
        assert recorder.record.call_count == 2

    def test_progress_indicator_shows_current_position_and_total(self) -> None:
        """Progress display shows e.g. '[3/28]' so the operator knows
        how far through the review they are."""
        session = ReviewSession(
            ranked_listings=[_make_ranked(external_id=f"p{i}") for i in range(28)],
            recorder=_make_recorder(),
        )
        progress = session.format_progress(current=3, total=28)
        assert "[3/28]" in progress

    def test_no_undecided_listings_prints_message_and_exits(self) -> None:
        """When all listings have decisions, the review prints a
        'nothing to review' message and exits cleanly."""
        r1 = _make_ranked(external_id="done-1")
        recorder = _make_recorder(decided_ids={"done-1"})
        session = ReviewSession(ranked_listings=[r1], recorder=recorder)

        undecided = session.undecided_listings()
        assert len(undecided) == 0
        # Message is handled by CLI; session just returns empty list

    def test_no_results_file_prints_message_and_exits(self) -> None:
        """When there are no ranked listings at all, the review reports
        'no results found' and exits cleanly."""
        session = ReviewSession(ranked_listings=[], recorder=_make_recorder())
        undecided = session.undecided_listings()
        assert len(undecided) == 0
